"""
Shogunet network action policy
==============================

The seam where ShogoCore's execution layer learns about network actions:
a union of action types precisely mirroring ShugoCore's ``policy.py``
structure, so a ShugoCore ``DecisionEngine`` that registers the network
handler can gate ``network_send`` / ``network_query`` / ``network_sync``
through its existing consent/approval brokers (side-effecting) and let
read-only intent through without consent (``network_list_agents`` /
``network_status``).

An agent's topics stay pinned to its own namespace: any network action the
engine proposes is only legitimate on ``/shugunet/{agent_id}/...``.
"""

NETWORK_ACTION_TYPES = {
    "network_send",          # side-effecting: message/request to a peer
    "network_query",         # side-effecting: mesh memory query (traffic)
    "network_sync",          # side-effecting: fact sync / digest exchange
}
NETWORK_READ_ACTION_TYPES = {
    "network_list_agents",   # read-only roster
    "network_status",        # read-only health/telemetry
}
KNOWN_ACTION_TYPES = (NETWORK_ACTION_TYPES | NETWORK_READ_ACTION_TYPES)

SIDE_EFFECTING_ACTION_TYPES = frozenset(NETWORK_ACTION_TYPES)

NETWORK_NAMESPACE = "/shugunet"


def network_topic(agent_id: str, tail: str) -> str:
    """Canonical topic inside the agent's own namespace."""
    tail = str(tail).strip("/")
    return f"{NETWORK_NAMESPACE}/{agent_id}/{tail}" if tail \
        else f"{NETWORK_NAMESPACE}/{agent_id}"


def is_network_topic(topic: str) -> bool:
    return str(topic).strip("/").startswith(NETWORK_NAMESPACE.lstrip("/") + "/")