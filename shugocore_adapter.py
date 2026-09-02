"""
ShugoCore-side adapter (drop-in, following the mobile-handler pattern)
=====================================================================

Copy this module into a ShugoCore checkout as ``shugonet_bridge.py`` (or
import it from the Shogunet checkout via ``SHUGOCORE_PATH``) to let a
ShugoCore ``DecisionEngine`` drive the Shogunet network stack exactly the
way it already drives the robotics and mobile handlers:

1. ``ShugonetExecutionHandler`` mirrors ``MobileExecutionHandler``: a
   ``handle(decision)`` entry that dispatches on ``action_type`` against a
   duck-typed ``shugonet_agent`` (the Shogunet-side runtime).
2. ``register_network_handlers`` augments ``policy.KNOWN_ACTION_TYPES`` and
   registers the handler on the engine's ``execution_layer`` -- the same
   seam ``decision_engine`` uses for robotics/mobile handlers.
3. ``attach_network_fallbacks`` copies Shogunet's network trigger severities
   into the engine's existing ``FallbackController`` so ``network_peer_lost``
   etc. latch the governor through the same deterministic path.

All ShugoCore imports are lazy and guarded, so importing this module inside
the Shogunet tree (for tests) never requires ShugoCore on sys.path.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NETWORK_ACTION_TYPES = {
    "network_send",          # side-effecting: message/request to a peer
    "network_query",         # side-effecting: mesh memory query
    "network_sync",          # side-effecting: fact sync / digest exchange
}
NETWORK_READ_ACTION_TYPES = {
    "network_list_agents",   # read-only roster
    "network_status",        # read-only health
}

NETWORK_FALLBACK_SEVERITIES = {
    "network_transport_exhausted": "pause",
    "network_peer_lost": "pause",
    "memory_sync_conflict_storm": "safe_state",
    "audit_chain_broken": "halt",
}

_NAMESPACE = "/shugunet"


def network_topic(agent_id: str, tail: str) -> str:
    """Canonical topic inside the agent's own namespace."""
    return f"{_NAMESPACE}/{agent_id}/{str(tail).strip('/')}"


class ShugonetExecutionHandler:
    """Execution-layer handler for Shogunet network actions.

    ``shugonet_agent`` is the Shogunet runtime on the same host: it must
    expose ``send(peer, topic, payload)``, ``query(...)``, ``sync(...)``,
    ``list_agents()`` and ``status()``. This mirrors the robotics handler's
    contract so the engine's approval/consent gates apply unchanged.
    """

    def __init__(self, shugonet_agent: Any):
        self.agent = shugonet_agent

    def handle(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        action_type = str(decision.get("action_type", ""))
        params = decision.get("params") or {}
        if action_type == "network_send":
            return self._send(params)
        if action_type == "network_query":
            return self._query(params)
        if action_type == "network_sync":
            return self._sync(params)
        if action_type == "network_list_agents":
            return {"status": "success", "action": "network_list_agents",
                    "agents": self.agent.list_agents()}
        if action_type == "network_status":
            return {"status": "success", "action": "network_status",
                    **self.agent.status()}
        return {"status": "refused",
                "reason": f"unknown network action '{action_type}'"}

# -- action dispatch ------------------------------------------------------

    def _send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        peer = str(params.get("peer", ""))
        topic = str(params.get("topic", ""))
        payload = params.get("payload")
        if not peer or not topic or payload is None:
            return {"status": "refused", "reason": "peer/topic/payload required"}
        result = self.agent.send(peer, topic, payload)
        return {"status": result.get("status", "success"),
                "action": "network_send", "peer": peer, **result}

    def _query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query")
        if not query:
            return {"status": "refused", "reason": "query text required"}
        result = self.agent.query(query, peers=params.get("peers"),
                                  top_k=params.get("top_k"))
        return {"status": "success", "action": "network_query",
                "results": result}

    def _sync(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = self.agent.sync(peer=params.get("peer"),
                                 since=params.get("since"))
        return {"status": "success", "action": "network_sync", **result}


def register_network_handlers(execution_layer: Any, shugonet_agent: Any,
                              policy_module: Optional[Any] = None,
                              action_types: Optional[List[str]] = None) -> None:
    """Register network action types + handler on a ShugoCore engine.

    ``execution_layer`` may be the engine's ``execution_layer`` or a test
    double exposing ``register_handler(action_type, fn)``. ``policy_module``
    defaults to ShugoCore's ``policy`` (imported lazily); pass a stub in
    tests to avoid requiring ShugoCore on sys.path.
    """
    handler = ShugonetExecutionHandler(shugonet_agent)
    types = action_types or sorted(NETWORK_ACTION_TYPES
                                   | NETWORK_READ_ACTION_TYPES)
    policy = policy_module
    if policy is None:
        try:
            import policy as _policy
            policy = _policy
        except Exception:
            policy = None
    if policy is not None:
        if not hasattr(policy, "KNOWN_ACTION_TYPES"):
            setattr(policy, "KNOWN_ACTION_TYPES", set())
        policy.KNOWN_ACTION_TYPES.update(types)
    for action_type in types:
        execution_layer.register_handler(action_type, handler.handle)
    logger.info("registered %d shugonet action handlers", len(types))


def attach_network_fallbacks(fallback_controller: Any) -> None:
    """Merge Shogunet network trigger severities into a ShugoCore
    ``FallbackController`` (or a duck-typed double)."""
    try:
        fallback_controller.severities.update(NETWORK_FALLBACK_SEVERITIES)
    except Exception as exc:
        logger.warning("network fallback severities not merged: %s", exc)