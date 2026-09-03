"""
Shogunet PostgreSQL fact mirror (persistence half of the memory mesh)
=====================================================================

Optional Tier-2 persistence for the codependent memory mesh: a drop-in
"backend" for ``MemorySyncNode`` that mirrors every mesh-applied fact into
PostgreSQL, using the same conventions as ShugoCore's ``pg_memory`` module
(the fleet-shared ``PgSemanticMemory``). Several hosts/agents pointing at
the same DSN therefore converge on one durable fact base, matching ShugoCore
v1.4.0's persistence half of the Shogunet memory mesh.

Contract (duck-typed, exactly what ``MemorySyncNode`` needs):

- ``upsert_fact(fact)``  -- fact is a store record dict (key, origin,
  fact_id, content, kind, salience, access_count, metadata, created_at,
  last_accessed, updated_at); the (origin, fact_id) pair is the identity.
- ``remove(key)``        -- tombstone propagation.
- ``count()``            -- current mirrored row count.

Fail-closed like ShugoCore: if psycopg2 is missing, the DSN is unreachable,
or the database is read-only, construction or the first write raises -- there
is NO silent stub mode. Install with ``pip install 'shugonet[postgres]'``.
"""

import json
import logging
import re
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2  # type: ignore
    _HAS_PSYCOPG = True
except ImportError:
    _HAS_PSYCOPG = False

# Table names cannot be bound as SQL parameters; validate before SQL.
_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

_INSTALL_HINT = (
    "psycopg2 is required for the Shogunet PostgreSQL fact mirror; "
    "install it with: pip install 'shugonet[postgres]'"
)


class PgFactMirror:
    """Mirror the memory mesh's facts into a PostgreSQL table.

    Thread-safe: a single connection guarded by an RLock, safe to attach to
    one ``MemorySyncNode`` (or share one mirror across nodes via an outer
    lock -- psycopg2 connections are not multiprocess-safe).
    """

    def __init__(self, dsn: str, table: str = "shugonet_facts",
                 connect_timeout_s: float = 5.0):
        if not _HAS_PSYCOPG:
            raise RuntimeError(_INSTALL_HINT)
        if not _PREFIX_RE.match(str(table or "")):
            raise ValueError(
                "table must match ^[a-z][a-z0-9_]{0,40}$ (identifier)")
        self.dsn = str(dsn)
        self.table = str(table)
        self._timeout_s = max(1.0, float(connect_timeout_s))
        self._lock = threading.RLock()
        self._conn = None
        self._connect()
        self._ensure_schema()

    # -- connection ------------------------------------------------------------

    def _connect(self) -> None:
        try:
            self._conn = psycopg2.connect(
                self.dsn, connect_timeout=int(self._timeout_s))
        except TypeError:      # older psycopg2 without the kwarg
            self._conn = psycopg2.connect(self.dsn)
        self._conn.autocommit = True

    def _ensure_schema(self) -> None:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            key            text PRIMARY KEY,
            origin         text NOT NULL,
            fact_id        bigint NOT NULL,
            content        text NOT NULL,
            kind           text NOT NULL DEFAULT 'fact',
            salience       real NOT NULL DEFAULT 1.0,
            access_count   integer NOT NULL DEFAULT 0,
            metadata       jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            created_at     double precision NOT NULL,
            last_accessed  double precision NOT NULL,
            updated_at     double precision NOT NULL
        );
        CREATE INDEX IF NOT EXISTS {self.table}_origin_idx
            ON {self.table} (origin);
        CREATE INDEX IF NOT EXISTS {self.table}_salience_idx
            ON {self.table} (salience DESC);
        """
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(ddl)
            except Exception:
                self._conn = None
                raise

    # -- mesh backend contract ---------------------------------------------------

    def upsert_fact(self, fact: Dict[str, Any]) -> None:
        """Mirror one mesh fact (``MemorySyncNode`` backend contract)."""
        fact = dict(fact or {})
        key = str(fact.get("key") or "")
        if not key:
            raise ValueError("fact.key required")
        metadata = fact.get("metadata")
        sql = f"""
        INSERT INTO {self.table}
            (key, origin, fact_id, content, kind, salience, access_count,
             metadata, created_at, last_accessed, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (key) DO UPDATE SET
            content       = EXCLUDED.content,
            kind          = EXCLUDED.kind,
            salience      = GREATEST({self.table}.salience, EXCLUDED.salience),
            access_count  = EXCLUDED.access_count,
            metadata      = EXCLUDED.metadata,
            last_accessed = EXCLUDED.last_accessed,
            updated_at    = EXCLUDED.updated_at
        """
        params = (
            key,
            str(fact.get("origin") or ""),
            int(fact.get("fact_id") or 0),
            str(fact.get("content") or ""),
            str(fact.get("kind") or "fact"),
            float(fact.get("salience") or 0.0),
            int(fact.get("access_count") or 0),
            json.dumps(metadata if isinstance(metadata, dict) else {}),
            float(fact.get("created_at") or 0.0),
            float(fact.get("last_accessed") or 0.0),
            float(fact.get("updated_at") or 0.0),
        )
        self._execute(sql, params)

    def remove(self, key: str) -> bool:
        """Tombstone propagation: delete one mirrored fact."""
        with self._lock:
            self._ensure_connected()
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self.table} WHERE key = %s", (str(key),))
                    return bool(cur.rowcount)
            except Exception:
                self._conn = None
                raise

    def count(self) -> int:
        with self._lock:
            self._ensure_connected()
            try:
                with self._conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {self.table}")
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
            except Exception:
                self._conn = None
                raise

    # -- internals ----------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._conn is not None and self._conn.closed == 0:
            return
        self._connect()

    def _execute(self, sql: str, params: tuple) -> None:
        with self._lock:
            self._ensure_connected()
            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
            except Exception:
                self._conn = None
                raise

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __len__(self) -> int:
        try:
            return self.count()
        except Exception:
            return 0
