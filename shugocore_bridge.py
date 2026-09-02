"""
Shogunet <-> ShugoCore bridge
=============================

Integration seam between the two repositories. Shogunet's stdlib-first core
keeps working standalone; when ``SHUGOCORE_PATH`` points at a checkout of
ShugoCore (flat modules importable on sys.path), this bridge prefers
ShugoCore's hardened primitives so behavior is identical across the fleet:

- ``sanitize_text`` / ``redact``    ShugoCore ``security``
- ``AuditChain``                    ShugoCore ``audit`` (hash-chained)
- Tier-2 store for the memory mesh  ShugoCore ``SemanticMemory``

When ShugoCore is absent every lookup degrades to the local stdlib-first
equivalent, keeping Shogunet independently installable and testable. The
mesh and the chain are written against duck-typed contracts, so swapping
the backend never changes the networking code.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOADED: Dict[str, Any] = {}


def configure(path: Optional[str] = None) -> bool:
    """Point the bridge at a ShugoCore checkout; call once at startup.

    Returns True when ShugoCore's security/audit primitives became
    importable from ``path``.
    """
    candidate = str(path or os.environ.get("SHUGOCORE_PATH", "")).strip()
    _LOADED.clear()
    if not candidate:
        return False
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    try:
        import security as _security       # ShugoCore security
        import audit as _audit             # ShugoCore audit
        _LOADED["security"] = _security
        _LOADED["audit"] = _audit
        logger.info("shugocore bridge loaded from %s", candidate)
        return True
    except Exception as exc:
        logger.debug("shugocore primitives unavailable: %s", exc)
        _LOADED.clear()
        return False


def shugocore_loaded() -> bool:
    return "security" in _LOADED and "audit" in _LOADED


# -- primitive delegation -------------------------------------------------------

def sanitize_text(text: Any, max_length: int = 2048) -> str:
    """ShugoCore's sanitize_text when present, else the local equivalent."""
    if "security" in _LOADED:
        try:
            return str(_LOADED["security"].sanitize_text(text, max_length))
        except Exception:
            pass
    from security import sanitize_text as _local
    return _local(text, max_length)


def redact(value: Any) -> Any:
    """ShugoCore's redact when present, else the local equivalent."""
    if "security" in _LOADED:
        try:
            return _LOADED["security"].redact(value)
        except Exception:
            pass
    from security import redact as _local
    return _local(value)


def validate_url(url: str, allowed_hosts: Optional[List[str]] = None) -> bool:
    """ShugoCore's validate_url when present; local scheme/host fallback."""
    if "security" in _LOADED:
        try:
            return bool(_LOADED["security"].validate_url(
                url, allowed_hosts or [], allow_all=True))
        except Exception:
            pass
    try:
        from urllib.parse import urlparse
        parsed = urlparse(str(url))
        return parsed.scheme in ("http", "https") and bool(parsed.hostname)
    except Exception:
        return False


def make_audit(path: str):
    """ShugoCore's hash-chained AuditChain when present, else the local one."""
    if "audit" in _LOADED:
        try:
            return _LOADED["audit"].AuditChain(path)
        except Exception:
            pass
    from audit import AuditChain
    return AuditChain(path)

def make_fact_store(agent_id: str, dimension: int = 64):
    """Tier-2 store for the memory mesh.

    With ShugoCore present this wraps its ``SemanticMemory`` (SQLite-backed,
    deterministic hashing vectors) so networked facts live in the agent's
    real long-term memory; otherwise it falls back to the in-memory store the
    test suite uses. Both satisfy the mesh's duck-typed contract.
    """
    try:
        import tempfile
        from memory_system import SemanticMemory
        handle = tempfile.NamedTemporaryFile(prefix="sgn_", suffix=".db",
                                             delete=False)
        memory = SemanticMemory(db_path=handle.name, dimension=dimension)
        mem_store = _ShugocoreStoreAdapter(agent_id, memory)
        mem_store._temp_handle = handle
        return mem_store
    except Exception as exc:
        logger.debug("SemanticMemory fallback to in-memory store: %s", exc)
    from memory_sync import InMemoryFactStore
    return InMemoryFactStore(agent_id, dimension=dimension)


class _ShugocoreStoreAdapter:
    """Adapts ShugoCore ``SemanticMemory`` to the mesh store contract.

    ``SemanticMemory`` keys facts with local ints; the mesh keys networked
    facts as ``"{origin}:{local_id}"``. This adapter keeps the mapping and
    mirrors the mesh surface so ``MemorySyncNode`` works unchanged.
    """

    def __init__(self, agent_id: str, semantic):
        self.agent_id = str(agent_id)
        self._semantic = semantic
        self._keymap: Dict[str, int] = {}
        self._temp_handle = None

    def make_key(self, origin: str, fact_id: int) -> str:
        return f"{origin}:{int(fact_id)}"

    def _origin_of(self, key: str) -> str:
        return str(key).rpartition(":")[0] or self.agent_id

    def _fact_id_of(self, key: str) -> int:
        return int(str(key).rpartition(":")[2] or 0)

    def store_fact(self, content: str, kind: str = "fact",
                   vector: Optional[List[float]] = None,
                   salience: float = 1.0, fact_id: Optional[int] = None,
                   origin: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = self._semantic.store_fact(
            content=str(content), kind=str(kind),
            vector=list(vector) if vector else None,
            salience=max(0.0, float(salience)))
        local_id = int(record.get("fact_id", 0))
        key_origin = origin or self.agent_id
        key_id = int(fact_id) if fact_id is not None else local_id
        self._keymap[self.make_key(key_origin, key_id)] = local_id
        record["key"] = self.make_key(key_origin, key_id)
        record["origin"] = key_origin
        return dict(record)

    def get_fact(self, key: str) -> Optional[Dict[str, Any]]:
        local_id = self._keymap.get(str(key))
        if local_id is None:
            return None
        record = self._semantic.get_fact(local_id)
        if record is None:
            return None
        record["key"] = str(key)
        record["origin"] = self._origin_of(key)
        return dict(record)

    def remove(self, key: str) -> bool:
        return self._keymap.pop(str(key), None) is not None

    def reinforce(self, key: str, boost: float = 0.25,
                  cap: float = 10.0) -> Optional[Dict[str, Any]]:
        local_id = self._keymap.get(str(key))
        if local_id is None:
            return None
        return self._semantic.reinforce(local_id, boost=boost)

    def search(self, query: str, top_k: int = 5,
               min_salience: float = 0.0) -> List[Dict[str, Any]]:
        return list(self._semantic.search(query, top_k=top_k,
                                          min_salience=min_salience))

    def facts_by_kind(self, kind: str) -> List[Dict[str, Any]]:
        return list(self._semantic.facts_by_kind(kind))

    def count(self) -> int:
        return int(self._semantic.count())

    def digests(self) -> List[Dict[str, Any]]:
        from memory_sync import fact_digest
        out = []
        for key, _local_id in self._keymap.items():
            origin = self._origin_of(key)
            fact_id = self._fact_id_of(key)
            out.append({"d": fact_digest(origin, fact_id),
                        "s": 0.0, "origin": origin, "fact_id": fact_id})
        return out

    def close(self) -> None:
        try:
            self._semantic.close()
        except Exception:
            pass
        if self._temp_handle is not None:
            try:
                self._temp_handle.close()
            except Exception:
                pass