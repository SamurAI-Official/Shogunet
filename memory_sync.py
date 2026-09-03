"""
Shogunet codependent memory mesh (N1 / N2)
==========================================

Syncs ShugoCore long-term memory (Tier 2) between paired agents so the fleet
learns *together* -- the "codependent" part of the memory layers:

- **Provenance.** Every networked fact is keyed by ``(origin_agent_id,
  local_fact_id)``. A foreign fact is read-only in the sense that only its
  origin teaches its content; peers reinforce and reference it.
- **Conflict resolution.** Content updates use last-writer-wins; salience
  merges as ``max`` with additive reinforcement; pruning propagates as
  tombstones so a forgotten memory cannot resurrect from a stale peer.
- **Reinforcement feedback loop.** When a peer retrieves and uses a
  fact, a tiny ``reinforce`` envelope flows back to the origin, boosting the
  origin's salience/access_count. Useful memories survive decay fleet-wide;
  unused ones decay everywhere. ShugoCore's deterministic hashing vectors
  make this work without any embedding service.
- **Anti-entropy digests.** Over constrained links (LoRa/BLE) peers exchange
  compact ``memory_digest`` frames -- digests of fact keys + salience -- and
  only pull actual fact content when a broadband link is available to carry
  it. The reinforcement/digest loop is the tier that runs on ~50-byte frames.

Memory invariants over the wire (unchanged from ShugoCore):
- Tier 0 / Tier 1 (scratchpad, episodic) are **N0: never transmitted**.
- Tier 2 semantic facts are **N1: shareable** via this module.
- Tier 3 is read-only: only **promotion proposals** travel, and applying one
  stays an operator-attributed privileged step at the receiving agent.
"""

import base64
import hashlib
import logging
import math
import re
import struct
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import protocol
from protocol import Envelope
from security import redact, sanitize_text

logger = logging.getLogger(__name__)

MAX_FACT_CONTENT = 4096
MAX_FACTS_PER_DIGEST = 32
MAX_VECTOR_DIM = 256


class SharingProfile(Enum):
    """Per-peer egress policy mirroring ShugoCore's roadmap item."""

    PRIVATE = "private"            # no sync in either direction
    REDACTED = "redacted-share"    # security.redact() on egress payloads
    FULL = "full-share"            # full content + vector


def hashed_embedding(text: str, dimension: int = 256,
                     normalize: bool = True) -> List[float]:
    """Deterministic hashing bag-of-words embedding, ShugoCore-compatible.

    This is an exact port of ShugoCore ``vector_db.hashed_embedding`` (which
    itself mirrors ``SemanticMemory._embed``): tokens are ``[a-z0-9]+`` runs,
    each token's sha256 picks one bucket (``digest % dimension``) and a sign
    (bit 128), L2-normalized. Vectors produced here are directly comparable
    with ShugoCore Tier-2 vectors -- mesh facts, ``PgSemanticMemory`` rows and
    local ``SemanticMemory`` facts all land in the same cosine space without
    any embedding service, on any transport.

    ``memory_sync._embed`` remains as a back-compat alias.
    """
    vector = [0.0] * int(dimension)
    for token in re.findall(r"[a-z0-9]+", str(text).lower()):
        digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        index = digest % int(dimension)
        sign = 1.0 if (digest >> 128) & 1 else -1.0
        vector[index] += sign
    if normalize:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
    return vector


# Back-compat alias: older call sites embed via _embed(text, dimension).
_embed = hashed_embedding


def _cosine(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n])) or 1.0
    nb = math.sqrt(sum(x * x for x in b[:n])) or 1.0
    return dot / (na * nb)


def fact_digest(origin: str, fact_id: object) -> str:
    """Anti-entropy digest of one networked fact key."""
    key = f"{sanitize_text(origin, 48)}:{int(fact_id) & 0xFFFFFFFF}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def pack_vector(vec: List[float]) -> str:
    """Pack a vector for the wire as base64 little-endian float32.

    The protocol bounds payload *lists* at 64 items (DoS bound), so a
    256-dim embedding cannot travel as a JSON list; packed base64 keeps it
    inside ``MAX_STR_VALUE`` (~1.4 KB for 256 dims) and lossless to float32.
    """
    clamped = [_clamp_float(x) for x in list(vec)[:MAX_VECTOR_DIM]]
    return base64.b64encode(
        struct.pack("<%df" % len(clamped), *clamped)).decode("ascii")


def unpack_vector(value: Any) -> Optional[List[float]]:
    """Unpack a ``pack_vector`` string; None when absent/corrupt."""
    if not isinstance(value, str) or not value:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) % 4 or len(raw) // 4 > MAX_VECTOR_DIM:
        return None
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num

class InMemoryFactStore:
    """Minimal Tier-2 stand-in used by the mesh and the test suite.

    Mirrors the subset of ShugoCore's ``SemanticMemory`` surface the mesh
    needs (search / store_fact / reinforce / facts_by_kind / count / get).
    ``shugocore_bridge`` later swaps this for the real store.
    """

    def __init__(self, agent_id: str, dimension: int = MAX_VECTOR_DIM):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        self.dimension = max(8, int(dimension))
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._next_local_id = 1
        self._lock = threading.RLock()

    # -- key helpers -------------------------------------------------------------

    def _local_key(self, fact_id: int) -> str:
        return f"{self.agent_id}:{int(fact_id)}"

    def _foreign_key(self, origin: str, fact_id: int) -> str:
        return f"{sanitize_text(origin, 48)}:{int(fact_id) & 0xFFFFFFFF}"

    def make_key(self, origin: str, fact_id: int) -> str:
        """Public key factory used by the mesh for cross-agent keys."""
        return self._foreign_key(origin, fact_id)

    # -- operations ------------------------------------------------------------

    def store_fact(self, content: str, kind: str = "fact",
                   vector: Optional[List[float]] = None,
                   salience: float = 1.0, fact_id: Optional[int] = None,
                   origin: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   created_at: Optional[float] = None,
                   last_accessed: Optional[float] = None) -> Dict[str, Any]:
        """Store a fact and return it. Local facts get an auto id; foreign
        facts (origin set) are stored under their origin's id."""
        content = sanitize_text(content, MAX_FACT_CONTENT)
        kind = sanitize_text(kind, 32) or "fact"
        if origin is None or origin == self.agent_id:
            origin = self.agent_id
            if fact_id is None:
                with self._lock:
                    fact_id = self._next_local_id
                    self._next_local_id += 1
            key = self._local_key(int(fact_id))
        else:
            key = self._foreign_key(origin, int(fact_id))
        vec = list(vector) if vector else _embed(content, self.dimension)
        now = time.time()
        # Networked facts carry their origin's clock so LWW ordering and
        # age-based decay survive the hop (bounded/sanitized on the way in).
        if created_at is not None and 0.0 <= float(created_at) <= now:
            created_at = float(created_at)
        else:
            created_at = now
        if last_accessed is not None and created_at <= float(last_accessed) <= now:
            last_accessed = float(last_accessed)
        else:
            last_accessed = now
        fact = {
            "key": key,
            "fact_id": int(fact_id),
            "origin": origin,
            "content": content,
            "kind": kind,
            "salience": max(0.0, float(salience)),
            "vector": vec,
            "metadata": dict(metadata or {}),
            "access_count": 0,
            # Timestamp columns mirror ShugoCore SemanticMemory / PgSemanticMemory
            # (created_at, last_accessed) so mesh facts round-trip into a
            # fleet-shared PostgreSQL Tier-2 store without field loss.
            "created_at": created_at,
            "last_accessed": last_accessed,
            "updated_at": now,
        }
        with self._lock:
            self._facts[key] = fact
        return dict(fact)

    def get_fact(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            fact = self._facts.get(str(key))
            if fact is not None:
                fact["last_accessed"] = time.time()
        return dict(fact) if fact else None

    def remove(self, key: str) -> bool:
        with self._lock:
            return self._facts.pop(str(key), None) is not None

    def reinforce(self, key: str, boost: float = 0.25,
                  cap: float = 10.0) -> Optional[Dict[str, Any]]:
        """Cross-agent reinforcement: raises salience and access_count."""
        with self._lock:
            fact = self._facts.get(str(key))
            if fact is None:
                return None
            fact["salience"] = min(float(cap),
                                   fact["salience"] + max(0.0, float(boost)))
            fact["access_count"] = int(fact.get("access_count", 0)) + 1
            fact["updated_at"] = time.time()
            return dict(fact)

    def search(self, query: str, top_k: int = 5,
               min_salience: float = 0.0) -> List[Dict[str, Any]]:
        q_vec = _embed(query, self.dimension)
        scored = []
        now = time.time()
        with self._lock:
            facts = list(self._facts.values())
        for fact in facts:
            if fact["salience"] < min_salience:
                continue
            score = _cosine(q_vec, fact["vector"]) * fact["salience"]
            scored.append((score, dict(fact)))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [fact for _score, fact in scored[: max(0, int(top_k))]]
        # Retrieval is access: mirror SemanticMemory's last_accessed touch so
        # decay/prune policies behave identically on mesh and local facts.
        with self._lock:
            for fact in results:
                stored = self._facts.get(fact["key"])
                if stored is not None:
                    stored["last_accessed"] = now
        return results

    def facts_by_kind(self, kind: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(f) for f in self._facts.values()
                    if f["kind"] == str(kind)]

    def count(self) -> int:
        with self._lock:
            return len(self._facts)

    def digests(self) -> List[Dict[str, Any]]:
        """Anti-entropy summary of every networked fact.

        Carries origin + fact_id so a receiving node can map a digest back to
        the fact content when it decides to pull what it is missing.
        """
        with self._lock:
            facts = list(self._facts.values())
        return [{"d": fact_digest(f["origin"], f["fact_id"]),
                 "s": round(f["salience"], 3),
                 "origin": f["origin"],
                 "fact_id": f["fact_id"]}
                for f in facts]

class MemorySyncNode:
    """Attaches a codependent memory mesh to a TransportChain.

    Proactive cycle (per consolidated ShugoCore fact):
        store_fact -> publish_fact -> memory_fact envelope on the chain
    Reactive cycle per inbound envelope:
        memory_fact  -> store / LWW update / additive reinforcement
        reinforce    -> origin applies salience + access_count boost
        tombstone    -> peers prune so forgotten memories cannot resurrect
        memory_digest-> anti-entropy: pull missing facts when a broadband
                        link can carry them
    """

    def __init__(self, agent_id: str, chain, store: InMemoryFactStore,
                 profiles: Optional[Dict[str, SharingProfile]] = None,
                 default_profile: SharingProfile = SharingProfile.FULL,
                 audit: Optional[Any] = None,
                 on_synced: Optional[Callable[[Dict[str, Any]], None]] = None,
                 backend: Optional[Any] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self.chain = chain
        self.store = store
        self.prot_agent_id = self.agent_id
        # Optional persistence half of the mesh (duck-typed like ShugoCore's
        # PgSemanticMemory): every applied/reinforced/removed mesh fact is
        # mirrored via backend.upsert_fact(fact) / backend.remove(key) so a
        # fleet-shared Tier-2 store stays consistent with the wire state.
        self.backend = backend
        self.default_profile = default_profile
        self.profiles = {k: (v if isinstance(v, SharingProfile)
                             else SharingProfile(str(v)))
                         for k, v in (profiles or {}).items()}
        self.audit = audit
        self.on_synced = on_synced
        self._tombstones: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = {"facts_sent": 0, "facts_received": 0,
                       "conflicts_resolved": 0, "reinforces_sent": 0,
                       "reinforces_applied": 0, "tombstones_sent": 0,
                       "tombstones_applied": 0, "digests_sent": 0,
                       "digests_received": 0, "facts_pulled": 0}
        chain.subscribe(self._on_envelope)

    # -- sharing profiles -----------------------------------------------------------

    def set_profile(self, peer_id: str, profile: SharingProfile) -> None:
        self.profiles[sanitize_text(peer_id, 48)] = profile

    def profile_for(self, peer_id: str) -> SharingProfile:
        return self.profiles.get(sanitize_text(peer_id, 48),
                                 self.default_profile)

    # -- backend mirror ---------------------------------------------------------

    def _backend_upsert(self, key: str) -> None:
        """Mirror one store fact into the persistence backend (if any)."""
        if self.backend is None:
            return
        fact = self.store.get_fact(key)
        if fact is None:
            return
        try:
            self.backend.upsert_fact(fact)
        except Exception:
            logger.warning("memory backend upsert failed", exc_info=True)

    def _backend_remove(self, key: str) -> None:
        if self.backend is None:
            return
        try:
            self.backend.remove(key)
        except Exception:
            logger.warning("memory backend remove failed", exc_info=True)

    # -- reactive: inbound -------------------------------------------------------

    def _on_envelope(self, env: protocol.Envelope, via: str) -> None:
        # Defense-in-depth cross-talk guard: the chain already filters, but a
        # node must never act on mail addressed to another agent even if it is
        # handed one directly (tests, future transports, relay races).
        if env.recipient not in ("*", self.prot_agent_id):
            return
        handler = {
            "memory_fact": self._on_fact,
            "reinforce": self._on_reinforce,
            "tombstone": self._on_tombstone,
            "memory_digest": self._on_digest,
        }.get(env.msg_type)
        if handler is None:
            return
        if self.profile_for(env.sender) is SharingProfile.PRIVATE:
            return
        try:
            handler(env)
        except Exception:
            logger.warning("memory_sync handler failed for %s",
                           env.msg_type, exc_info=True)

    def _on_fact(self, env: protocol.Envelope) -> None:
        payload = env.payload
        origin = sanitize_text(payload.get("origin") or env.sender, 48)
        fact_id = payload.get("fact_id")
        content = payload.get("content")
        if origin in (None, "") or not isinstance(fact_id, int) \
                or content is None:
            self._audit("memory_sync_malformed",
                        {"msg_id": env.msg_id, "kind": "memory_fact"})
            return
        if origin == self.prot_agent_id:
            return                      # a loop: ignore our own facts
        key = self.store.make_key(origin, fact_id)
        existing = self.store.get_fact(key)
        incoming_metadata = payload.get("metadata")
        incoming_metadata = dict(incoming_metadata) \
            if isinstance(incoming_metadata, dict) and len(incoming_metadata) <= 16 \
            else {}
        try:
            incoming_created = float(payload.get("created_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            incoming_created = 0.0
        # Prefer the packed vector; tolerate the legacy JSON-list form from
        # older peers; otherwise the receiver re-embeds from content (the
        # embedding is deterministic and fleet-shared, so this is lossless).
        incoming_vector = unpack_vector(payload.get("vector_b64"))
        if incoming_vector is None:
            legacy = payload.get("vector")
            if isinstance(legacy, list) and legacy \
                    and len(legacy) <= MAX_PAYLOAD_ITEMS:
                incoming_vector = [_clamp_float(x) for x in legacy]
        incoming = {
            "content": sanitize_text(content, MAX_FACT_CONTENT),
            "kind": sanitize_text(payload.get("kind", "fact"), 32),
            "vector": incoming_vector,
            "salience": max(0.0, float(payload.get("salience", 1.0))),
        }
        if existing is None:
            self.store.store_fact(content=incoming["content"],
                                  kind=incoming["kind"],
                                  vector=incoming["vector"],
                                  salience=incoming["salience"],
                                  fact_id=fact_id, origin=origin,
                                  metadata=incoming_metadata,
                                  created_at=incoming_created or None)
            self._stats["facts_received"] += 1
            self._backend_upsert(key)
            fact = self.store.get_fact(key)
            if self.on_synced is not None:
                try:
                    self.on_synced(fact)
                except Exception:
                    logger.warning("on_synced callback failed", exc_info=True)
            return
        if existing["content"] != incoming["content"]:
            # Last-writer-wins on content; salience merges as max. The fact
            # keeps its original creation clock (identity), like SemanticMemory.
            self.store.store_fact(content=incoming["content"],
                                  kind=incoming["kind"],
                                  vector=incoming["vector"],
                                  salience=max(existing["salience"],
                                               incoming["salience"]),
                                  fact_id=fact_id, origin=origin,
                                  metadata=incoming_metadata or existing.get("metadata"),
                                  created_at=existing.get("created_at"))
            self._stats["conflicts_resolved"] += 1
            self._backend_upsert(key)
            self._audit("memory_conflict_resolved",
                        {"key": key, "origin": origin})
            return
        # Same content: additive reinforcement -- the memory proved durable
        # across the fleet, so it ranks a little higher everywhere.
        self.store.reinforce(key, boost=0.05)
        self._backend_upsert(key)

    def _on_reinforce(self, env: protocol.Envelope) -> None:
        origin = sanitize_text(env.payload.get("origin") or env.sender, 48)
        fact_id = env.payload.get("fact_id")
        amount = env.payload.get("amount")
        if origin != self.prot_agent_id or not isinstance(fact_id, int):
            return                      # only the origin applies reinforces
        key = self.store.make_key(origin, fact_id)
        try:
            boost = max(0.0, min(2.0, float(amount))) if amount is not None \
                else 0.25
        except (TypeError, ValueError):
            boost = 0.25
        if self.store.reinforce(key, boost=boost) is not None:
            self._stats["reinforces_applied"] += 1
            self._backend_upsert(key)

    def _on_tombstone(self, env: protocol.Envelope) -> None:
        origin = sanitize_text(env.payload.get("origin") or env.sender, 48)
        fact_id = env.payload.get("fact_id")
        if not isinstance(fact_id, int) or origin in (None, ""):
            return
        if origin == self.prot_agent_id:
            return
        key = self.store.make_key(origin, fact_id)
        if self.store.get_fact(key) is not None:
            self.store.remove(key)
            self._stats["tombstones_applied"] += 1
            self._backend_remove(key)
        # Remember so a stale peer cannot resurrect it during this session.
        self._tombstones[key] = time.monotonic()

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.append(event_type, payload)
        except Exception:
            pass

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    # -- proactive: outbound -----------------------------------------------------

    def publish_fact(self, fact: Dict[str, Any],
                     peers: Optional[List[str]] = None,
                     qos: str = "best_effort") -> None:
        """Publish one local fact to peers (respecting sharing profiles).

        ``fact`` is a store record: requires ``fact_id``, ``content``,
        ``kind``, ``salience``, ``vector``. Facts we learned from peers are
        never republished (no relay storm): ``origin`` must be us.

        With ``peers=None`` the fact is broadcast to every reachable peer
        (recipient ``"*"``), applying the default profile; with an explicit
        peer list each peer is addressed individually and gets its own
        profile's treatment.
        """
        origin = sanitize_text(fact.get("origin") or self.prot_agent_id, 48)
        if origin != self.prot_agent_id:
            return
        fact_id = fact.get("fact_id")
        content = fact.get("content")
        if not isinstance(fact_id, int) or not content:
            return
        payload = {
            "fact_id": fact_id,
            "content": sanitize_text(content, MAX_FACT_CONTENT),
            "kind": sanitize_text(fact.get("kind", "fact"), 32),
            "salience": max(0.0, float(fact.get("salience", 1.0))),
            "origin": origin,
        }
        created_at = fact.get("created_at")
        if isinstance(created_at, (int, float)) and float(created_at) > 0:
            payload["created_at"] = float(created_at)
        metadata = fact.get("metadata")
        if isinstance(metadata, dict) and 0 < len(metadata) <= 16:
            payload["metadata"] = metadata
        vec = fact.get("vector")
        if isinstance(vec, list) and vec:
            # Packed base64 float32: a 256-dim ShugoCore-parity embedding is
            # ~1.4 KB, inside MAX_STR_VALUE. The receiver may also simply
            # re-embed (deterministic shared algorithm) when this is absent.
            payload["vector_b64"] = pack_vector(vec)
        if peers is not None:
            for peer in peers:
                profile = self.profile_for(peer)
                if profile is SharingProfile.PRIVATE:
                    continue
                out_payload = redact(payload) \
                    if profile is SharingProfile.REDACTED else payload
                env = Envelope(msg_id=protocol.new_msg_id(),
                               msg_type="memory_fact",
                               sender=self.prot_agent_id,
                               recipient=sanitize_text(peer, 48),
                               topic=f"/shugunet/{self.prot_agent_id}/memory",
                               payload=out_payload)
                self.chain.send(env, qos=qos)
                self._stats["facts_sent"] += 1
            return
        # Default: broadcast to the fleet.
        if self.default_profile is SharingProfile.PRIVATE:
            return
        out_payload = redact(payload) \
            if self.default_profile is SharingProfile.REDACTED else payload
        env = Envelope(msg_id=protocol.new_msg_id(),
                       msg_type="memory_fact",
                       sender=self.prot_agent_id,
                       recipient="*",
                       topic=f"/shugunet/{self.prot_agent_id}/memory",
                       payload=out_payload)
        if self.chain.send(env, qos=qos).ok:
            self._stats["facts_sent"] += 1

    def send_reinforce(self, peer: str, origin: str, fact_id: int,
                       amount: float = 0.25) -> None:
        """Tell ``peer`` that one of its facts proved useful here."""
        env = Envelope(msg_id=protocol.new_msg_id(), msg_type="reinforce",
                       sender=self.prot_agent_id, recipient=str(peer),
                       topic=f"/shugunet/{self.prot_agent_id}/memory",
                       payload={"origin": sanitize_text(origin, 48),
                                "fact_id": int(fact_id),
                                "amount": max(0.0, min(2.0, float(amount)))})
        if self.chain.send(env).ok:
            self._stats["reinforces_sent"] += 1

    def publish_tombstone(self, origin: str, fact_id: int,
                          reason: str = "pruned") -> None:
        """Broadcast that a fact was forgotten, so no peer resurrects it."""
        env = Envelope(msg_id=protocol.new_msg_id(), msg_type="tombstone",
                       sender=self.prot_agent_id, recipient="*",
                       topic=f"/shugunet/{self.prot_agent_id}/memory",
                       payload={"origin": sanitize_text(origin, 48),
                                "fact_id": int(fact_id),
                                "reason": sanitize_text(reason, 120)})
        if self.chain.send(env).ok:
            self._stats["tombstones_sent"] += 1

    def send_digests(self, peer: str) -> None:
        """Anti-entropy summary to one peer (runs fine on LoRa frames)."""
        env = Envelope(msg_id=protocol.new_msg_id(), msg_type="memory_digest",
                       sender=self.prot_agent_id, recipient=str(peer),
                       topic=f"/shugunet/{self.prot_agent_id}/memory",
                       payload={"digests": self.store.digests()})
        if self.chain.send(env).ok:
            self._stats["digests_sent"] += 1

    def _on_digest(self, env: protocol.Envelope) -> None:
        """Anti-entropy: a peer sent its fact digests, so we push them the
        local facts they are missing (addressed to that peer).

        Digests are P1 and ride fine on LoRa; the fact content is also P1 but
        the chain's class/chunk rules only let it travel when a medium that
        can carry it is online -- so the digest loop runs everywhere and the
        content pull happens whenever bandwidth is available.
        """
        peer = env.sender
        incoming = env.payload.get("digests") or []
        peer_digests = {str(entry["d"]) for entry in incoming
                        if isinstance(entry, dict) and entry.get("d")}
        missing = []
        for entry in self.store.digests():
            if entry["origin"] != self.prot_agent_id:
                continue                       # only teach facts we own
            if entry["d"] in peer_digests:
                continue
            fact = self.store.get_fact(
                self.store.make_key(entry["origin"], entry["fact_id"]))
            if fact is not None:
                missing.append(fact)
        missing.sort(key=lambda f: f["salience"], reverse=True)
        for fact in missing[: MAX_FACTS_PER_DIGEST]:
            self.publish_fact(fact, peers=[peer])
            self._stats["facts_pulled"] += 1
        self._stats["digests_received"] += 1

# [SGN:CONT]