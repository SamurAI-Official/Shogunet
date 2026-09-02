"""
Shogunet agent registry
=======================

Trust anchor of the fleet, mirroring ShugoCore's mobile-node security model:

- **Pairing = consent.** Only operator-allowlisted agent_ids are accepted;
  a pairing grant carries a TTL (default 12h) and is audited.
- **Topic ACL.** A paired agent may only publish on its own namespace,
  ``/shugunet/{agent_id}/{tail}``; inbound data on any other topic is
  refused and audited. Memory, task and control traffic all flow through
  this single seam.
- **Realm.** Each agent declares ``sim`` or ``phys`` so simulation fleets and
  physical fleets collaborate over the same protocol while downstream policy
  (e.g. sim-to-real memory sharing) can gate on it.
- **Liveness.** Heartbeats refresh per-agent liveness; stale agents go quiet
  without being unpaired (reconnection re-pairs seamlessly while the TTL
  holds).
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from security import sanitize_text

DEFAULT_PAIRING_TTL_HOURS = 12.0
HEARTBEAT_TIMEOUT_S = 30.0


def parse_shugonet_topic(topic: str) -> Optional[Tuple[str, str]]:
    """Split ``/shugunet/{agent_id}/{tail}``; None outside the namespace or
    when any segment is empty."""
    parts = str(topic or "").strip("/").split("/")
    if (len(parts) == 3 and parts[0] == "shugunet"
            and parts[1] and parts[2]):
        return parts[1], parts[2]
    return None


class AgentRegistry:
    """Operator-managed pairing of ShugoCore agents, with TTLs and ACLs."""

    def __init__(self, audit: Optional[Any] = None,
                 pairing_ttl_hours: float = DEFAULT_PAIRING_TTL_HOURS,
                 heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT_S):
        self.audit = audit
        self.pairing_ttl_hours = max(0.01, float(pairing_ttl_hours))
        self.heartbeat_timeout_s = max(1.0, float(heartbeat_timeout_s))
        self._paired: Dict[str, Dict[str, Any]] = {}
        self._last_heartbeat: Dict[str, float] = {}
        # Reentrant: roster/ACL helpers call is_paired() while holding this.
        self._lock = threading.RLock()

    # -- pairing (consent) ---------------------------------------------------

    def pair(self, agent_id: str, manifest: Optional[Dict[str, Any]] = None,
             paired_by: str = "operator") -> Dict[str, Any]:
        """Grant a pairing; only this call admits an agent to the fleet."""
        agent = sanitize_text(agent_id, 48).strip()
        if not agent:
            raise ValueError("agent_id required")
        realm = "phys"
        clean_manifest: Dict[str, Any] = {}
        if isinstance(manifest, dict):
            realm = sanitize_text(manifest.get("realm", "phys"), 8) or "phys"
            clean_manifest = dict(manifest)
        entry = {
            "agent_id": agent,
            "manifest": clean_manifest,
            "realm": realm,
            "paired_by": sanitize_text(paired_by, 120),
            "paired_at": time.time(),
            "expires_at": time.time() + self.pairing_ttl_hours * 3600.0,
        }
        with self._lock:
            self._paired[agent] = entry
            self._last_heartbeat[agent] = time.monotonic()
        self._audit("agent_paired", {"agent_id": agent,
                                     "paired_by": entry["paired_by"],
                                     "realm": realm})
        return dict(entry)

    def unpair(self, agent_id: str) -> bool:
        with self._lock:
            removed = self._paired.pop(str(agent_id), None)
            self._last_heartbeat.pop(str(agent_id), None)
        if removed is not None:
            self._audit("agent_unpaired", {"agent_id": str(agent_id)})
        return removed is not None

    def is_paired(self, agent_id: str) -> bool:
        """Paired and inside the TTL (lazy expiry)."""
        agent = str(agent_id)
        with self._lock:
            entry = self._paired.get(agent)
            if entry is None:
                return False
            if time.time() > entry["expires_at"]:
                self._paired.pop(agent, None)
                self._last_heartbeat.pop(agent, None)
                expired = True
            else:
                expired = False
        if expired:
            self._audit("agent_pairing_expired", {"agent_id": agent})
        return not expired

    # -- liveness ------------------------------------------------------------

    def heartbeat(self, agent_id: str) -> bool:
        """Record liveness; False when the agent is not currently paired."""
        agent = str(agent_id)
        now = time.monotonic()
        with self._lock:
            if not self.is_paired(agent):
                return False
            self._last_heartbeat[agent] = now
        return True

    def alive(self, agent_id: str) -> bool:
        with self._lock:
            if not self.is_paired(str(agent_id)):
                return False
            last = self._last_heartbeat.get(str(agent_id))
        return (last is not None
                and time.monotonic() - last <= self.heartbeat_timeout_s)

    # -- roster ----------------------------------------------------------------

    def list_agents(self) -> List[Dict[str, Any]]:
        with self._lock:
            agents = [dict(entry) for entry in self._paired.values()
                      if self.is_paired(entry["agent_id"])]
        for entry in agents:
            entry["alive"] = self.alive(entry["agent_id"])
        return agents

    def manifest(self, agent_id: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._paired.get(str(agent_id))
        return dict(entry["manifest"]) if entry else {}

    def realm(self, agent_id: str) -> Optional[str]:
        with self._lock:
            entry = self._paired.get(str(agent_id))
        return entry["realm"] if entry else None

    # -- topic ACL ------------------------------------------------------------

    def can_publish(self, agent_id: str, topic: str) -> bool:
        """An agent publishes only under its own ``/shugunet/{id}/{tail}``."""
        agent = str(agent_id)
        parsed = parse_shugonet_topic(topic)
        return (bool(agent) and parsed is not None and parsed[0] == agent
                and self.is_paired(agent))

    def can_accept(self, sender_id: str, topic: str) -> bool:
        """Inbound data is accepted only from a paired sender publishing in
        the sender's own namespace -- cross-namespace injection is
        unreachable by construction."""
        sender = str(sender_id)
        parsed = parse_shugonet_topic(topic)
        return (self.is_paired(sender) and parsed is not None
                and parsed[0] == sender)

    # -- internals ------------------------------------------------------------

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.append(event_type, payload)
        except Exception:
            pass

