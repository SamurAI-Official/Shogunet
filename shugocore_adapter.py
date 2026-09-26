"""
ShugoCore-side adapter (drop-in, following the mobile-handler pattern)
=====================================================================

Compatible with ShugoCore v1.20.0+. Copy this module into a ShugoCore
checkout as ``shugonet_bridge.py`` (or import it from the Shogunet checkout
via ``SHUGOCORE_PATH``) to let a ShugoCore ``DecisionEngine`` drive the
Shogunet network stack exactly the way it already drives the robotics and
mobile handlers:

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
    "network_send",              # side-effecting: message/request to a peer
    "network_query",             # side-effecting: mesh memory query
    "network_sync",              # side-effecting: fact sync / digest exchange
    "network_spatial_observe",   # side-effecting: publish 3D spatial observation
    "network_nrr_render",        # side-effecting: dispatch NRR frame render
    "network_nrr_scene",         # side-effecting: dispatch NRR motion/scene perception
}
NETWORK_READ_ACTION_TYPES = {
    "network_list_agents",       # read-only roster
    "network_status",            # read-only health
    "network_spatial_query",     # read-only 3D spatial search
    "network_fleet_map",         # read-only consolidated 3D fleet map
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
        if action_type == "network_spatial_observe":
            return self._spatial_observe(params)
        if action_type == "network_spatial_query":
            return self._spatial_query(params)
        if action_type == "network_fleet_map":
            return self._fleet_map(params)
        if action_type == "network_nrr_render":
            return self._nrr_render(params)
        if action_type == "network_nrr_scene":
            return self._nrr_scene(params)
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

    def _spatial_observe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.agent, "publish_observation"):
            return {"status": "refused", "reason": "runtime does not support spatial observations"}
        entity_id = params.get("entity_id")
        x = params.get("x")
        y = params.get("y")
        z = params.get("z", 0.0)
        if entity_id is None or x is None or y is None:
            return {"status": "refused", "reason": "entity_id, x, y required"}
        res = self.agent.publish_observation(
            entity_id=str(entity_id), x=float(x), y=float(y), z=float(z),
            confidence=float(params.get("confidence", 1.0)),
            label=str(params.get("label", "")),
            frame_id=str(params.get("frame_id", "world")),
        )
        return {"status": "success" if res.get("status") == "success" else "failed",
                "action": "network_spatial_observe", **res}

    def _spatial_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.agent, "query_nearby"):
            return {"status": "refused", "reason": "runtime does not support spatial queries"}
        x = params.get("x")
        y = params.get("y")
        z = params.get("z", 0.0)
        radius = params.get("radius")
        if x is None or y is None or radius is None:
            return {"status": "refused", "reason": "x, y, radius required"}
        results = self.agent.query_nearby(float(x), float(y), float(z), float(radius))
        return {"status": "success", "action": "network_spatial_query", "results": results}

    def _fleet_map(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.agent, "get_fleet_map"):
            return {"status": "refused", "reason": "runtime does not support fleet map"}
        fleet_map = self.agent.get_fleet_map()
        return {"status": "success", "action": "network_fleet_map", "fleet_map": fleet_map}

    def _nrr_render(self, params: Dict[str, Any]) -> Dict[str, Any]:
        peer = params.get("peer")
        desc = params.get("descriptor")
        if not peer or not desc:
            return {"status": "refused", "reason": "peer and descriptor required"}
        if not hasattr(self.agent, "render_frame"):
            return {"status": "refused", "reason": "runtime does not support nrr rendering"}
        res = self.agent.render_frame(str(peer), dict(desc))
        return {"status": res.get("status", "success"), "action": "network_nrr_render", **res}

    def _nrr_scene(self, params: Dict[str, Any]) -> Dict[str, Any]:
        peer = params.get("peer")
        req = params.get("motion_request")
        if not peer or not req:
            return {"status": "refused", "reason": "peer and motion_request required"}
        if not hasattr(self.agent, "query_scene"):
            return {"status": "refused", "reason": "runtime does not support nrr scene perception"}
        res = self.agent.query_scene(str(peer), dict(req))
        return {"status": res.get("status", "success"), "action": "network_nrr_scene", **res}


def _load_shugocore_policy() -> Optional[Any]:
    """Resolve ShugoCore's ``policy`` module, or None.

    Shogunet has its own ``policy`` module, so a bare ``import policy`` can
    bind the wrong one and leave the execution layer rejecting our action
    types. The bridge can load ShugoCore's copy by explicit file path; fall
    back to a bare import (correct inside a ShugoCore process, where no other
    ``policy`` exists).
    """
    try:
        from shugocore_bridge import load_shugocore_module
    except Exception:
        load_shugocore_module = None
    if load_shugocore_module is not None:
        module = load_shugocore_module("policy")
        if module is not None:
            return module
    try:
        import policy
        return policy
    except Exception:
        return None


def _merge_into(policy: Any, attr: str, values: Any) -> None:
    """Add ``values`` to a policy action-type set **in place**.

    Consumers that did ``from policy import NETWORK_ACTION_TYPES`` hold a
    reference to this very object -- ``execution_layer`` does exactly that and
    re-reads it on every ``register_handler`` call -- so rebinding the
    attribute would leave them validating a stale set. An immutable
    (frozenset) set has to be rebound; that is logged because an
    already-imported consumer may not observe it.
    """
    current = getattr(policy, attr, None)
    if current is None:
        setattr(policy, attr, set(values))
        return
    if isinstance(current, frozenset):
        logger.warning("policy.%s is a frozenset; rebinding (imported "
                       "consumers may keep the old set)", attr)
        setattr(policy, attr, set(current) | set(values))
        return
    try:
        current.update(values)
    except AttributeError:      # pragma: no cover - exotic container
        setattr(policy, attr, set(current) | set(values))


def register_network_handlers(execution_layer: Any, shugonet_agent: Any,
                              policy_module: Optional[Any] = None,
                              action_types: Optional[List[str]] = None
                              ) -> List[str]:
    """Register network action types + handler on a ShugoCore engine.

    ``execution_layer`` may be the engine's ``execution_layer`` or a test
    double exposing ``register_handler(action_type, fn)``. ``policy_module``
    defaults to ShugoCore's ``policy`` (resolved by path when the bridge can
    reach a checkout); pass a stub in tests to avoid requiring ShugoCore.

    Returns the action types that were successfully registered. A type the
    execution layer refuses is logged and skipped rather than aborting the
    whole batch; only a total failure raises, so a caller running
    ``attach_network_fallbacks`` straight afterwards still gets its severities.
    """
    handler = ShugonetExecutionHandler(shugonet_agent)
    types = list(action_types or sorted(NETWORK_ACTION_TYPES
                                        | NETWORK_READ_ACTION_TYPES))
    policy = policy_module if policy_module is not None \
        else _load_shugocore_policy()
    if policy is not None:
        # ``ExecutionLayer.register_handler`` validates against *these* two
        # sets (not KNOWN_ACTION_TYPES), so the side-effecting and read-only
        # classes must learn the spatial/NRR types or registration raises
        # ValueError. Consent gating is unaffected: the decision engine reads
        # NETWORK_ACTION_TYPES straight from this module at import time.
        _merge_into(policy, "NETWORK_ACTION_TYPES",
                    NETWORK_ACTION_TYPES & set(types))
        _merge_into(policy, "NETWORK_READ_ACTION_TYPES",
                    NETWORK_READ_ACTION_TYPES & set(types))
        _merge_into(policy, "KNOWN_ACTION_TYPES", types)
    registered: List[str] = []
    rejected: List[str] = []
    for action_type in types:
        try:
            execution_layer.register_handler(action_type, handler.handle)
            registered.append(action_type)
        except Exception as exc:
            rejected.append("%s (%s)" % (action_type, exc))
    if rejected:
        logger.warning("shugonet action handlers rejected by the execution "
                       "layer: %s", "; ".join(rejected))
    if not registered:
        raise RuntimeError("no shugonet action handlers could be "
                           "registered: " + "; ".join(rejected))
    logger.info("registered %d shugonet action handlers", len(registered))
    return registered


def attach_network_fallbacks(fallback_controller: Any) -> None:
    """Merge Shogunet network trigger severities into a ShugoCore
    ``FallbackController`` (or a duck-typed double)."""
    try:
        fallback_controller.severities.update(NETWORK_FALLBACK_SEVERITIES)
    except Exception as exc:
        logger.warning("network fallback severities not merged: %s", exc)