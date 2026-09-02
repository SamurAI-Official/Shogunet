"""
Shogunet mesh memory query
==========================

Fan a semantic query out to online paired peers and merge the results into
a single provenance-labelled answer for the deciding engine.

Lifecycle
---------
- ``query(text, peers)`` sends one ``memory_query`` envelope per peer
  (addressed, P2 -- requires an IP-capable link; constrained media carry the
  digests/facts instead) and collects ``memory_response`` envelopes for up to
  ``timeout_s``. Polling is driven in-process, so the test suite is
  deterministic: no background threads race the assertion.
- Every returned result is labelled with its origin agent
  ``(origin, fact_id)`` so downstream decisions can reason about *where* a
  memory came from -- and so the reinforcement loop can later send a
  ``reinforce`` back to the right agent (codependency).
- Results merge by fact key, score as ``similarity * salience``, and are
  capped at ``top_k``. Sharing profiles are honoured: PRIVATE peers are never
  queried and never answer.

Standalone responder: the same object auto-answers ``memory_query``
envelopes from its own store, so a fleet of nodes each running a
``MeshQuery`` attached to their chain can serve each other without any extra
server component.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import protocol
from protocol import Envelope
from security import sanitize_text
from memory_sync import InMemoryFactStore, SharingProfile

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1.5
DEFAULT_TOP_K = 8
MAX_QUERY_TEXT = 512


class MeshQuery:
    """Fan-out memory queries over a TransportChain, with auto-respond."""

    def __init__(self, agent_id: str, chain, store: InMemoryFactStore,
                 profiles: Optional[Dict[str, SharingProfile]] = None,
                 default_profile: SharingProfile = SharingProfile.FULL):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self.chain = chain
        self.store = store
        self.default_profile = default_profile
        self.profiles = {k: (v if isinstance(v, SharingProfile)
                             else SharingProfile(str(v)))
                         for k, v in (profiles or {}).items()}
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stats = {"queries_sent": 0, "replies_received": 0,
                       "queries_answered": 0}
        chain.subscribe(self._on_envelope)

    # -- profiles -----------------------------------------------------------

    def set_profile(self, peer_id: str, profile: SharingProfile) -> None:
        self.profiles[sanitize_text(peer_id, 48)] = profile

    def profile_for(self, peer_id: str) -> SharingProfile:
        return self.profiles.get(sanitize_text(peer_id, 48),
                                 self.default_profile)

    # -- client side ----------------------------------------------------------

    def query(self, query: str, peers: Optional[List[str]] = None,
              top_k: int = DEFAULT_TOP_K,
              timeout_s: float = DEFAULT_TIMEOUT_S,
              poller: Optional[Callable[[], None]] = None) -> List[Dict[str, Any]]:
        """Fan out one query; returns merged, provenance-labelled results.

        ``poller`` drives the receive path: in a single-process fleet the
        caller supplies a poller that pumps every agent's chain so remote
        responders are serviced inside the wait window. Defaults to polling
        only our own chain.
        """
        query = sanitize_text(query, MAX_QUERY_TEXT)
        peers = [p for p in (peers or [])
                 if sanitize_text(p, 48) and sanitize_text(p, 48) != self.agent_id
                 and self.profile_for(p) is not SharingProfile.PRIVATE]
        if not query or not peers:
            return []
        if poller is None:
            def poller() -> None:
                self.chain.poll(0.05)
        qid = protocol.new_msg_id()
        pending = {"results": [], "keys": set(), "replies": 0,
                   "expected": len(peers)}
        with self._lock:
            self._pending[qid] = pending
        for peer in peers:
            env = Envelope(msg_id=qid, msg_type="memory_query",
                           sender=self.agent_id,
                           recipient=sanitize_text(peer, 48),
                           topic=f"/shugunet/{self.agent_id}/memory",
                           payload={"query": query,
                                    "top_k": max(1, int(top_k)),
                                    "correlation": qid})
            self.chain.send(env)
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        while time.monotonic() < deadline:
            poller()
            with self._lock:
                current = self._pending.get(qid)
                if current is None or current["replies"] >= current["expected"]:
                    break
        with self._lock:
            current = self._pending.pop(qid, None)
        if current is None:
            return []
        results = sorted(current["results"],
                         key=lambda r: float(r.get("score", 0.0)),
                         reverse=True)
        self._stats["queries_sent"] += 1
        return results[: max(1, int(top_k))]

# -- resident peer -----------------------------------------------------------

    def _on_envelope(self, env: protocol.Envelope, via: str) -> None:
        # Only act on queries/answers addressed to us or the fleet.
        if env.recipient not in ("*", self.agent_id):
            return
        if env.msg_type == "memory_query":
            self._answer(env)
        elif env.msg_type == "memory_response":
            self._collect(env)

    def _answer(self, env: protocol.Envelope) -> None:
        """Standalone responder: search our Tier-2 store and reply."""
        if (env.sender == self.agent_id
                or self.profile_for(env.sender) is SharingProfile.PRIVATE):
            return
        if env.recipient not in ("*", self.agent_id):
            return
        query = sanitize_text(env.payload.get("query", ""), MAX_QUERY_TEXT)
        if not query:
            return
        top_k = int(env.payload.get("top_k", DEFAULT_TOP_K))
        correlation = env.payload.get("correlation") or env.msg_id
        hits = self.store.search(query, top_k=top_k)
        results = [{
            "origin": sanitize_text(h["origin"], 48),
            "fact_id": int(h["fact_id"]),
            "content": sanitize_text(h["content"], 2048),
            "kind": sanitize_text(h["kind"], 32),
            "salience": round(float(h["salience"]), 3),
        } for h in hits]
        reply = Envelope(msg_id=protocol.new_msg_id(),
                         msg_type="memory_response",
                         sender=self.agent_id, recipient=env.sender,
                         topic=f"/shugunet/{self.agent_id}/memory",
                         payload={"correlation": correlation,
                                  "results": results})
        if self.chain.send(reply).ok:
            self._stats["queries_answered"] += 1

    def _collect(self, env: protocol.Envelope) -> None:
        correlation = env.payload.get("correlation")
        incoming = env.payload.get("results") or []
        if not isinstance(correlation, int) or not isinstance(incoming, list):
            return
        with self._lock:
            pending = self._pending.get(correlation)
            if pending is None:
                return
            for item in incoming:
                if not isinstance(item, dict):
                    continue
                origin = sanitize_text(item.get("origin", ""), 48)
                fact_id = item.get("fact_id")
                if not origin or not isinstance(fact_id, int):
                    continue
                key = f"{origin}:{fact_id}"
                if key in pending["keys"]:
                    continue
                pending["keys"].add(key)
                pending["results"].append({
                    "origin": origin,
                    "fact_id": int(fact_id),
                    "content": sanitize_text(item.get("content", ""), 2048),
                    "kind": sanitize_text(item.get("kind", "fact"), 32),
                    "salience": float(item.get("salience", 1.0)),
                    "score": float(item.get("salience", 1.0)),
                    "peer": sanitize_text(env.sender, 48),
                })
            pending["replies"] += 1
        self._stats["replies_received"] += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)