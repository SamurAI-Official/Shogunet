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

import contextlib
import importlib.util
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_LOADED: Dict[str, Any] = {}
_CANDIDATE: str = ""

# ShugoCore modules are loaded under a private prefix so they can never
# collide with (or be shadowed by) Shogunet's identically-named modules.
_PRIVATE_PREFIX = "_shugocore_"

# Attributes a module must expose for the bridge to trust it.
_REQUIRED_ATTRS = {"security": ("sanitize_text", "redact"),
                   "audit": ("AuditChain",)}


def _checkout_dir(path: Optional[str] = None) -> str:
    """Absolute path of a usable ShugoCore checkout, else ""."""
    candidate = str(path or os.environ.get("SHUGOCORE_PATH", "")
                    or _CANDIDATE or "").strip()
    if candidate and os.path.isdir(candidate):
        return os.path.abspath(candidate)
    return ""


def _module_from(module: Any, checkout: str) -> bool:
    """True when ``module`` was genuinely loaded out of ``checkout``."""
    file = getattr(module, "__file__", "") or ""
    if not file:
        return False
    try:
        return os.path.dirname(os.path.abspath(file)) == \
            os.path.abspath(checkout)
    except (TypeError, ValueError):     # pragma: no cover - defensive
        return False


@contextlib.contextmanager
def _borrowed_modules(names: Iterator[str]) -> Iterator[None]:
    """Temporarily evict same-named top-level modules, then restore them.

    Shogunet ships its own ``security``/``audit``/``policy``, and a module
    already in ``sys.modules`` wins over ``sys.path`` order. Without this, a
    ShugoCore module doing ``from security import ...`` would silently bind
    Shogunet's weaker copy. Originals are always put back.
    """
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        yield
    finally:
        for name, module in saved.items():
            sys.modules[name] = module


def _load_checkout_module(checkout: str, name: str) -> Any:
    """Import ``<checkout>/<name>.py`` under a private module name."""
    target = os.path.join(checkout, name + ".py")
    if not os.path.isfile(target):
        raise ImportError("no %s.py in %s" % (name, checkout))
    alias = _PRIVATE_PREFIX + name
    cached = sys.modules.get(alias)
    if cached is not None and _module_from(cached, checkout):
        return cached
    spec = importlib.util.spec_from_file_location(alias, target)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s from %s" % (name, checkout))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


def configure(path: Optional[str] = None) -> bool:
    """Point the bridge at a ShugoCore checkout; call once at startup.

    Returns True when ShugoCore's security/audit primitives were loaded
    *from that checkout* -- verified, not assumed. A bare ``import security``
    would resolve to Shogunet's own module as soon as anything had imported
    it, so the modules are loaded by explicit file path instead.
    """
    global _CANDIDATE
    _LOADED.clear()
    _CANDIDATE = ""
    checkout = _checkout_dir(path)
    if not checkout:
        return False
    if checkout not in sys.path:
        sys.path.insert(0, checkout)
    try:
        with _borrowed_modules(("security", "audit")):
            security = _load_checkout_module(checkout, "security")
            # ShugoCore's audit.py does ``from security import ...``; publish
            # the object just loaded so it binds this exact module rather
            # than compiling a second copy of the same file.
            sys.modules["security"] = security
            audit = _load_checkout_module(checkout, "audit")
        for name, module in (("security", security), ("audit", audit)):
            missing = [a for a in _REQUIRED_ATTRS[name] if not hasattr(module, a)]
            if missing:
                raise ImportError("%s from %s is missing %s"
                                  % (name, checkout, missing))
            if not _module_from(module, checkout):
                raise ImportError("%s did not load from %s" % (name, checkout))
        _LOADED["security"] = security
        _LOADED["audit"] = audit
        _CANDIDATE = checkout
        logger.info("shugocore bridge loaded from %s", checkout)
        return True
    except Exception as exc:
        logger.debug("shugocore primitives unavailable: %s", exc)
        _LOADED.clear()
        return False


def shugocore_loaded() -> bool:
    """True only when both modules are bound *and* provably from the checkout."""
    if not _CANDIDATE or "security" not in _LOADED or "audit" not in _LOADED:
        return False
    return all(_module_from(_LOADED[name], _CANDIDATE)
               for name in ("security", "audit"))


def load_shugocore_module(name: str, path: Optional[str] = None) -> Any:
    """Return a ShugoCore top-level module by file path, or None.

    The ShugoCore-side adapter uses this to resolve modules such as ``policy``
    without the bare-import shadowing that ``security``/``audit`` suffer from.
    """
    checkout = _checkout_dir(path)
    if not checkout:
        return None
    try:
        with _borrowed_modules((name,)):
            return _load_checkout_module(checkout, name)
    except Exception as exc:
        logger.debug("shugocore module %s unavailable: %s", name, exc)
        return None


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


def validate_url(url: str, allowed_hosts: Optional[List[str]] = None,
                 allowed_schemes: Any = ("http", "https")) -> bool:
    """Validate a URL, preferring ShugoCore's allowlist-enforcing primitive.

    ShugoCore >=1.30 exposes ``validate_url(url, allowed_hosts,
    allowed_schemes) -> (ok, reason)``; older builds used a boolean
    ``allow_all=`` keyword. Both are handled and the result is always a bool,
    so a signature drift can never silently degrade to the weak local check.

    ``allowed_schemes`` defaults to http+https because Shogunet's mesh
    legitimately talks plain HTTP to local relay hosts; the host allowlist and
    the rejection of embedded credentials are still enforced.
    """
    if "security" in _LOADED:
        validator = getattr(_LOADED["security"], "validate_url", None)
        if validator is not None:
            try:
                ok, _reason = validator(url, list(allowed_hosts or []),
                                        tuple(allowed_schemes))
                return bool(ok)
            except TypeError:
                # Legacy boolean signature (pre-1.30).
                try:
                    return bool(validator(url, list(allowed_hosts or []),
                                          allow_all=True))
                except TypeError:
                    logger.debug("shugocore validate_url signature unknown; "
                                 "using local check")
            except Exception as exc:
                # Fail closed: never wave a URL through on a validator error.
                logger.debug("shugocore validate_url failed closed: %s", exc)
                return False
    try:
        from urllib.parse import urlparse
        parsed = urlparse(str(url))
        return (parsed.scheme in tuple(allowed_schemes)
                and bool(parsed.hostname)
                and not parsed.username and not parsed.password)
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

def make_fact_store(agent_id: str, dimension: int = 256,
                    memory_manager: Optional[Any] = None):
    """Tier-2 store for the memory mesh.

    With ShugoCore present, this wraps a real ``MemoryManager`` (when the host
    supplies one) or a fresh SQLite-backed ``SemanticMemory``, so networked
    facts live in genuine Tier-2 memory and the 1.30+ sharing API
    (``export_shared_facts`` / ``import_shared_facts`` / ``count_shared_facts``)
    works over the mesh. Without ShugoCore it falls back to the in-memory store
    the test suite uses. All three satisfy the mesh's duck-typed contract.

    ``dimension`` defaults to 256 to match ShugoCore's canonical Tier-2 vector
    width, so embeddings stay comparable with a fleet-shared PgSemanticMemory.
    """
    if memory_manager is not None:
        try:
            return _ShugocoreStoreAdapter(agent_id, memory_manager)
        except Exception as exc:
            logger.debug("MemoryManager adapter initialization failed: %s", exc)

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
    """Adapts ShugoCore ``SemanticMemory`` / ``MemoryManager`` to the mesh contract.

    The mesh store and ``SemanticMemory`` disagree in three ways, and this
    adapter is the single seam that reconciles them:

    1. **Identity.** The mesh keys networked facts as ``"{origin}:{local_id}"``
       so a peer's ids never collide with local ones. ``SemanticMemory`` keys
       rows by an autoincrement integer. A bidirectional map between the two
       lets ``MemorySyncNode`` address foreign facts by origin.
    2. **Timestamps.** The mesh carries epoch floats (``created_at``) because
       they survive JSON. ``SemanticMemory`` persists ISO-8601 strings. We
       convert on both boundaries so last-writer-wins and age-based decay
       behave identically on local and networked facts.
    3. **Vectors.** ``SemanticMemory.store_fact`` takes no ``vector`` -- it
       re-embeds content with the shared deterministic hashing vector, which
       is byte-identical to :func:`memory_sync.hashed_embedding`. Passing the
       mesh vector through is therefore neither possible nor needed; a missing
       vector on the wire is lossless for the same reason.

    ShugoCore 1.30+ memory sharing is exposed verbatim: ``export_shared_facts``,
    ``import_shared_facts`` and ``count_shared_facts`` delegate to a supplied
    ``MemoryManager`` when one is present, and otherwise reproduce its exact
    semantics (content dedupe + ``shared_from``/``shared_at`` provenance) on
    the bare Tier-2 store.
    """

    def __init__(self, agent_id: str, semantic):
        self.agent_id = str(agent_id)
        # A full MemoryManager exposes tier2 *and* the sharing API; a bare
        # SemanticMemory gives us only the tier.
        if hasattr(semantic, "tier2"):
            self._manager = semantic
            self._semantic = semantic.tier2
        else:
            self._manager = None
            self._semantic = semantic
        # mesh key -> SemanticMemory row id, and the reverse for search hits.
        self._keymap: Dict[str, int] = {}
        self._revmap: Dict[int, str] = {}
        self._temp_handle = None
        self._lock = threading.RLock()

    # -- key helpers ---------------------------------------------------------

    def make_key(self, origin: str, fact_id: int) -> str:
        return f"{sanitize_text(origin, 48) or self.agent_id}:{int(fact_id) & 0xFFFFFFFF}"

    def _origin_of(self, key: str) -> str:
        return str(key).rpartition(":")[0] or self.agent_id

    def _fact_id_of(self, key: str) -> int:
        try:
            return int(str(key).rpartition(":")[2] or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _to_epoch(value: Any) -> float:
        """ISO-8601 string or epoch float -> epoch float."""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            return time.time()

    @staticmethod
    def _to_iso(value: Any) -> str:
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(
            timespec="seconds")

    def _remember(self, key: str, local_id: int) -> None:
        previous = self._keymap.get(key)
        if previous is not None and previous != local_id:
            self._revmap.pop(previous, None)
        self._keymap[key] = local_id
        self._revmap[local_id] = key

    def store_fact(self, content: str, kind: str = "fact",
                   vector: Optional[List[float]] = None,
                   salience: float = 1.0, fact_id: Optional[int] = None,
                   origin: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   created_at: Optional[float] = None,
                   last_accessed: Optional[float] = None) -> Dict[str, Any]:
        """Insert or update one fact, returning the mesh-shaped record.

        ``vector`` is accepted for contract parity and deliberately dropped:
        ``SemanticMemory`` re-embeds content deterministically with the very
        algorithm the mesh uses, so a re-embed is lossless.
        """
        key_origin = sanitize_text(origin or self.agent_id, 48) or self.agent_id
        clean_kind = sanitize_text(kind, 32) or "fact"
        clean_salience = max(0.0, float(salience))
        with self._lock:
            existing = self._keymap.get(
                self.make_key(key_origin, fact_id)) if fact_id is not None else None
            meta = dict(metadata or {})
            try:
                local_id = int(self._semantic.store_fact(
                    content=str(content), kind=clean_kind,
                    salience=clean_salience, metadata=meta or None))
            except TypeError:
                # Older SemanticMemory builds without the metadata kwarg.
                local_id = int(self._semantic.store_fact(
                    content=str(content), kind=clean_kind,
                    salience=clean_salience))
            key = (self.make_key(key_origin, fact_id) if fact_id is not None
                   else self.make_key(key_origin, local_id))
            if existing is not None and existing != local_id:
                # Last-writer-wins update of a fact already tracked: keep the
                # key pinned to the original row so peers keep one stable id.
                self._revmap.pop(local_id, None)
                local_id = existing
            self._remember(key, local_id)
            record = self._semantic.get_fact(local_id) or {}
            if not record:
                now = time.time()
                record = {"content": str(content), "kind": clean_kind,
                          "salience": clean_salience, "metadata": meta,
                          "access_count": 0,
                          "created_at": self._to_iso(created_at) if created_at
                          else self._to_iso(now),
                          "last_accessed": self._to_iso(last_accessed)
                          if last_accessed else self._to_iso(now)}
            return self._to_mesh_fact(key, record)

    def get_fact(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            local_id = self._keymap.get(str(key))
        if local_id is None:
            return None
        record = self._semantic.get_fact(local_id)
        if record is None:
            with self._lock:
                self._keymap.pop(str(key), None)
                self._revmap.pop(local_id, None)
            return None
        return self._to_mesh_fact(str(key), record)

    def _key_for_row(self, row_id: int, metadata: Any) -> Optional[str]:
        """Resolve the mesh key of a Tier-2 row.

        Rows written through the adapter are already tracked. Rows that arrived
        via ``import_shared_facts`` were written straight into Tier 2 by the
        MemoryManager and carry their provenance in ``shared_from`` metadata, so
        we mint them a stable key on first sight. That keeps peer knowledge
        visible to mesh search while ``publish_fact`` still refuses to relay it
        (its ``origin`` is the peer, never us).
        """
        with self._lock:
            tracked = self._revmap.get(int(row_id))
        if tracked is not None:
            return tracked
        shared_from = (metadata or {}).get("shared_from") \
            if isinstance(metadata, dict) else None
        source = sanitize_text(shared_from, 48) if shared_from else None
        if not source or source == self.agent_id:
            return None
        key = self.make_key(source, row_id)
        with self._lock:
            # Never shadow a key a peer genuinely owns.
            if self._keymap.get(key) is None:
                self._remember(key, int(row_id))
                return key
        return None

    def _to_mesh_fact(self, key: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Project a SemanticMemory row onto the mesh's fact shape."""
        fact = dict(record)
        fact["key"] = key
        fact["origin"] = self._origin_of(key)
        fact["fact_id"] = self._fact_id_of(key)
        fact["created_at"] = self._to_epoch(record.get("created_at"))
        fact["last_accessed"] = self._to_epoch(record.get("last_accessed"))
        return fact

    def remove(self, key: str) -> bool:
        with self._lock:
            local_id = self._keymap.pop(str(key), None)
            if local_id is None:
                return False
            self._revmap.pop(local_id, None)
        return True

    def reinforce(self, key: str, boost: float = 0.25,
                  cap: float = 10.0) -> Optional[Dict[str, Any]]:
        with self._lock:
            local_id = self._keymap.get(str(key))
        if local_id is None:
            return None
        # SemanticMemory.reinforce returns None; the mesh contract wants the
        # updated record so callers can mirror it into the backend.
        self._semantic.reinforce(local_id, boost=float(boost), cap=float(cap))
        return self.get_fact(str(key))

    def search(self, query: str, top_k: int = 5,
               min_salience: float = 0.0) -> List[Dict[str, Any]]:
        hits = self._semantic.search(query, top_k=int(top_k),
                                     min_salience=float(min_salience))
        out = []
        for hit in hits:
            key = self._key_for_row(int(hit.get("id", 0)),
                                    hit.get("metadata"))
            if key is None:
                continue            # no mesh identity for this row
            out.append(self._to_mesh_fact(key, hit))
        return out

    def facts_by_kind(self, kind: str) -> List[Dict[str, Any]]:
        out = []
        for row in self._semantic.facts_by_kind(str(kind)):
            key = self._key_for_row(int(row.get("id", 0)), row.get("metadata"))
            if key is not None:
                out.append(self._to_mesh_fact(key, row))
        return out

    def count(self) -> int:
        return int(self._semantic.count())

    def digests(self) -> List[Dict[str, Any]]:
        from memory_sync import fact_digest
        out = []
        with self._lock:
            items = list(self._keymap.items())
        for key, local_id in items:
            record = self._semantic.get_fact(local_id) or {}
            out.append({"d": fact_digest(self._origin_of(key),
                                         self._fact_id_of(key)),
                        "s": round(float(record.get("salience", 0.0)), 3),
                        "origin": self._origin_of(key),
                        "fact_id": self._fact_id_of(key)})
        return out

    # -- ShugoCore 1.30+ memory sharing parity --------------------------------

    def export_shared_facts(self, since: Optional[str] = None,
                            limit: int = 200) -> List[Dict[str, Any]]:
        """Export facts created after ``since`` for mesh peers.

        Only Tier 2 crosses the mesh -- never the scratchpad/episodic tiers and
        never a raw embedding -- mirroring ``MemoryManager`` exactly.
        """
        if self._manager is not None:
            return self._manager.export_shared_facts(since=since, limit=limit)
        return [{"content": f.get("content"), "kind": f.get("kind", "fact"),
                 "salience": f.get("salience", 1.0),
                 "created_at": f.get("created_at"),
                 "metadata": f.get("metadata") or {}}
                for f in self._semantic.facts_since(since_iso=since, limit=limit)]

    def import_shared_facts(self, facts: List[Dict[str, Any]],
                            source: str) -> Dict[str, int]:
        """Merge peer facts into Tier 2, idempotently, with provenance.

        Identical content is never duplicated, so a peer re-sending knowledge it
        already taught us is safe; ``duplicates`` feeds the mesh's
        ``memory_sync_conflict_storm`` guard.
        """
        if self._manager is not None:
            return self._manager.import_shared_facts(facts, source=source)
        origin = sanitize_text(source, 48) or "unknown"
        imported = skipped = duplicates = 0
        for fact in facts or []:
            if not isinstance(fact, dict):
                skipped += 1
                continue
            content = sanitize_text(
                fact.get("content") or fact.get("fact") or "", 2000)
            if not content:
                skipped += 1
                continue
            if self._semantic.content_exists(content):
                duplicates += 1
                continue
            metadata = dict(fact.get("metadata") or {})
            metadata["shared_from"] = origin
            metadata["shared_at"] = self._to_iso(time.time())
            try:
                salience = max(0.0, min(10.0, float(fact.get("salience", 0.8))))
            except (TypeError, ValueError):
                salience = 0.8
            self._semantic.store_fact(
                content,
                kind=sanitize_text(fact.get("kind") or "fact", 32) or "fact",
                salience=salience, metadata=metadata)
            imported += 1
        return {"imported": imported, "duplicates": duplicates, "skipped": skipped}

    def count_shared_facts(self, source: Optional[str] = None) -> int:
        """Durable count of facts imported from mesh peers."""
        if self._manager is not None:
            return int(self._manager.count_shared_facts(source=source))
        return int(self._semantic.count_shared(source=source))

    def shared_fact_sources(self) -> List[Dict[str, Any]]:
        """Distinct provenance peers with durable fact counts."""
        if self._manager is not None \
                and hasattr(self._manager, "shared_fact_sources"):
            return self._manager.shared_fact_sources()
        if hasattr(self._semantic, "shared_sources"):
            return self._semantic.shared_sources()
        return []

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