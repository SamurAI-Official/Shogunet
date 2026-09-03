"""
ShugonetHost -- server host for a fleet of ShugoCore agents
===========================================================

One process that owns the fleet's trust and transport plane so ShugoCore
agents only need a client runtime to join:

- ``AgentRegistry``   pairing = consent; TTLs; topic ACLs; liveness
- ``RelayHub``        HTTP rendezvous for NAT'd / cellular agents
- ``TCPTransport``    LAN listener whose handshake admits only paired agents
- ``MemorySyncNode``  the host's own Tier-2 store, the mesh's seed/backup
- ``MeshQuery``       the host answers fleet memory queries
- ``AuditChain``      every admission/refusal/pairing/latch event is chained
- ``NetworkFallbackController``  deterministic peer-loss/pause latching

Routing (hub-and-spoke): agents connect to the host; the host forwards
addressed mail to its recipient and fans broadcasts out as addressed copies
to every other paired agent (excluding the original sender). Host-addressed
mail -- heartbeats, mesh queries, memory broadcasts -- is processed locally by
the host's own chain handlers.

Trust model: pairing is enforced twice -- at TCP admission (handshake hook)
and again on every host-processed/forwarded envelope (the relay path has no
registry gate of its own, so the host re-checks the sender there).
"""

import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from agent_registry import AgentRegistry
from audit import AuditChain
from fallbacks import NetworkFallbackController
from memory_sync import InMemoryFactStore, MemorySyncNode
from mesh_query import MeshQuery
from protocol import Envelope, ProtocolError, decode, encode, new_msg_id
from relay_server import RelayHub, run_relay
from tcp_transport import TCPTransport
from transport_fallback import TransportChain

logger = logging.getLogger(__name__)

DEFAULT_HOST_ID = "shugonet-host"


class _HostGovernor:
    """Duck-typed governor the NetworkFallbackController latches through."""

    def __init__(self, host: "ShugonetHost"):
        self._host = host

    def pause(self, reason: str) -> None:
        self._host._latch("paused", reason)

    def safe_state(self, reason: str) -> None:
        self._host._latch("safe_state", reason)

    def halt(self, reason: str) -> None:
        self._host._latch("halted", reason)

    def resume(self, resumed_by: str = "") -> None:
        self._host._latch("normal", "resumed", resumed_by)


class ShugonetHost:
    """Server host: admit paired ShugoCore agents, route their traffic,
    seed the memory mesh, and expose an operator surface."""

    def __init__(self, agent_id: str = DEFAULT_HOST_ID,
                 tcp_host: str = "127.0.0.1", tcp_port: int = 0,
                 relay_port: int = 0, registry: Optional[AgentRegistry] = None,
                 audit: Optional[AuditChain] = None,
                 store: Optional[InMemoryFactStore] = None,
                 hub_max_messages: int = 1024,
                 heartbeat_timeout_s: float = 30.0,
                 dashboard_port: int = 0,
                 dashboard_bind: str = "127.0.0.1",
                 dashboard_token: Optional[str] = None,
                 dashboard_static_dir: Optional[str] = None):
        self.agent_id = agent_id
        self.tcp_host = tcp_host
        self.tcp_port_wanted = int(tcp_port)
        self.relay_port_wanted = int(relay_port)
        self.registry = registry or AgentRegistry(
            heartbeat_timeout_s=heartbeat_timeout_s)
        self.audit = audit or AuditChain(
            os.path.join(tempfile.mkdtemp(), "host-audit.jsonl"))
        self.store = store or InMemoryFactStore(agent_id)
        self.hub = RelayHub(max_messages=hub_max_messages)
        self.governor = _HostGovernor(self)
        self.fallback = NetworkFallbackController(self.governor,
                                                  audit=self.audit)
        self.registry.audit = self.audit
        self._running = False
        self._stop = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._relay_server = None
        self._relay_thread = None
        self._relay_url: Optional[str] = None
        self._mode = "normal"
        self._mode_reason = ""
        self._lost_report_cooldown_s = 10.0
        self._last_lost_report: Dict[str, float] = {}
        self._lost_reported: set = set()
        self._forwarded = 0
        self._refused_unpaired = 0
        # Built in start(): the host's own transports + mesh participants.
        self.tcp: Optional[TCPTransport] = None
        self.relay = None
        self.chain: Optional[Any] = None
        self.mesh: Optional[MeshQuery] = None
        # Operator plane: the fleet dashboard (stdlib HTTP + SSE + compiled SPA).
        self.dashboard_port = max(0, int(dashboard_port))
        self.dashboard_bind = str(dashboard_bind or "127.0.0.1")
        self.dashboard_token = dashboard_token
        self.dashboard_static_dir = dashboard_static_dir
        self._dashboard: Optional[Any] = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        # Relay rendezvous first: its port is advertised to joining agents.
        self._relay_server, self._relay_thread = run_relay(
            self.hub, self.tcp_host, self.relay_port_wanted)
        relay_port = self._relay_server.server_address[1]
        self._relay_url = f"http://{self.tcp_host}:{relay_port}"
        # TCP listener; admission consults the registry at every handshake.
        from relay_transport import RelayTransport
        import version
        self.tcp = TCPTransport(
            self.agent_id, listen=True, listen_host=self.tcp_host,
            listen_port=self.tcp_port_wanted,
            admission_check=self.registry.is_paired, audit=self.audit,
            on_connection_lost=self._on_peer_connection_lost,
            version_check=version.is_compatible)
        self.relay = RelayTransport(self.agent_id, hub_url=self._relay_url)
        # Raw-frame routing: the forwarder sees every inbound frame from every
        # transport and re-sends addressed mail / fans broadcasts out. The
        # chain separately processes host-addressed mail (mesh, heartbeats).
        self.chain = TransportChain(
            self.agent_id, [self.tcp, self.relay], audit=self.audit)
        # Router fires BEFORE the recipient guard so the host can forward
        # addressed mail and fan broadcasts (sees ALL envelopes, not just
        # those addressed to the host).
        self.chain.register_router(self._on_raw_frame)
        self.chain.subscribe(self._on_host_mail)
        # The host is a mesh participant: it stores every fleet fact (seed)
        # and answers memory queries against its own Tier-2 store.
        self.mesh = MeshQuery(self.agent_id, self.chain, self.store)
        self.tcp.start()
        self.relay.start()
        self._running = True
        self._stop.clear()
        self._loop_thread = threading.Thread(target=self._loop,
                                             name="sgn-host-loop", daemon=True)
        self._loop_thread.start()
        # Operator plane after the data plane: the dashboard serves the host's
        # status surface and streams audit events; a failure to bind it must
        # not take the fleet down.
        if self.dashboard_port > 0:
            try:
                from dashboard import DashboardServer
                self._dashboard = DashboardServer(
                    self, port=self.dashboard_port, bind=self.dashboard_bind,
                    token=self.dashboard_token,
                    static_dir=self.dashboard_static_dir)
                self._dashboard.start()
            except Exception as exc:
                logger.warning("dashboard failed to start: %s", exc)
                self._dashboard = None
        self._audit("host_started", {
            "agent_id": self.agent_id,
            "tcp_port": self.tcp_port, "relay_port": relay_port})

    def stop(self) -> None:
        if not self._running and self._loop_thread is None:
            return
        self._running = False
        self._stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None
        if self.tcp is not None:
            self.tcp.stop()
        if self.relay is not None:
            self.relay.stop()
        if self._relay_server is not None:
            self._relay_server.shutdown()
            self._relay_server.server_close()
            self._relay_server = None
        self._relay_thread = None
        self._relay_url = None
        if self._dashboard is not None:
            try:
                self._dashboard.stop()
            except Exception:
                pass
            self._dashboard = None
        self._audit("host_stopped", {"agent_id": self.agent_id})

    def _loop(self) -> None:
        """Host network loop: drains inbound mail; periodic liveness scan."""
        last_liveness = 0.0
        while not self._stop.is_set():
            try:
                if self.chain is not None:
                    self.chain.poll(0.05)
                now = time.monotonic()
                if now - last_liveness >= 1.0:
                    self.check_liveness()
                    last_liveness = now
            except Exception:
                logger.warning("host loop error", exc_info=True)
                self._stop.wait(0.1)

    @property
    def tcp_port(self) -> int:
        return self.tcp.local_port if self.tcp is not None \
            else self.tcp_port_wanted

    @property
    def relay_url(self) -> Optional[str]:
        return self._relay_url

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def mode_reason(self) -> str:
        return self._mode_reason


    # -- routing ---------------------------------------------------------------

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            self.audit.append(event_type, payload)
        except Exception:
            pass

    def _latch(self, mode: str, reason: str, resumed_by: str = "") -> None:
        self._mode = mode
        self._mode_reason = reason
        self._audit("host_mode_changed",
                    {"mode": mode, "reason": reason, "resumed_by": resumed_by})

    def _on_raw_frame(self, env: Envelope, via: str) -> bool:
        """Route inbound envelope: forward addressed mail, fan broadcasts.

        Routers fire BEFORE the recipient guard with (env, via). Returns True
        if the frame was consumed (forwarded/dropped), False if it should still
        be processed by local handlers.
        """
        if not self.registry.is_paired(env.sender):
            self._audit("unpaired_sender_dropped",
                        {"sender": str(env.sender)[:48]})
            return True  # consumed (dropped)
        if env.recipient == self.agent_id:
            return False  # for the host; let handlers process it
        if env.recipient == "*":
            for agent in self._paired_agents_excluding(env.sender):
                self._forward_to_agent(env, agent)
            return False  # host also processes broadcasts locally
        # Addressed to another agent: forward and consume (don't process locally)
        self._forward_to_agent(env, env.recipient)
        return True

    def _forward_to_agent(self, env: Envelope, recipient: str) -> None:
        """Forward an envelope to a specific agent via the chain."""
        try:
            frame = encode(env)
            self.chain.send_raw(recipient, frame)
        except Exception:
            logger.warning("forward to %s failed", recipient, exc_info=True)

    def _paired_agents_excluding(self, exclude: str) -> List[str]:
        with self.registry._lock:
            return [aid for aid in self.registry._paired.keys()
                    if aid != exclude and self.registry.is_paired(aid)]

    def _on_host_mail(self, env: Any, via: str) -> None:
        if env.msg_type == "heartbeat":
            self.registry.heartbeat(env.sender)

    def _on_peer_connection_lost(self, agent_id: str) -> None:
        """Callback from TCPTransport when a paired agent's connection drops."""
        if self.registry.is_paired(agent_id):
            self.fallback.report_violation(
                "network_peer_lost",
                f"agent {agent_id} connection lost")

    def check_liveness(self) -> None:
        for agent_info in self.registry.list_agents():
            aid = agent_info["agent_id"]
            if not agent_info.get("alive", False):
                last = self._last_lost_report.get(aid, 0.0)
                now = time.monotonic()
                if now - last >= self._lost_report_cooldown_s:
                    self._last_lost_report[aid] = now
                    self.fallback.report_violation(
                        "network_peer_lost", f"agent {aid} stale")

    # -- operator surface -----------------------------------------------------

    def status(self) -> Dict[str, Any]:
        import version
        roster = self.registry.list_agents()
        # Client versions reported via the TCP handshake (relay-only peers
        # report through their join manifest, surfaced below).
        peer_versions = self.tcp.peer_versions() if self.tcp is not None else {}
        relay_manifests = {}
        try:
            relay_manifests = {aid: info.get("manifest") or {}
                               for aid, info in self.hub.agents().items()}
        except Exception:
            pass
        for entry in roster:
            aid = entry["agent_id"]
            client_version = peer_versions.get(aid) \
                or str(relay_manifests.get(aid, {}).get("shugonet_version", "")
                       or "unknown")
            entry["client_version"] = client_version
        return {
            "agent_id": self.agent_id,
            "shugonet_version": version.VERSION,
            "protocol_version": version.PROTOCOL_VERSION,
            "mode": self.mode,
            "mode_reason": self.mode_reason,
            "tcp_port": self.tcp_port,
            "relay_url": self._relay_url,
            "dashboard_url": self._dashboard.url if self._dashboard else None,
            "roster": roster,
            "paired_count": len(roster),
            "alive_count": sum(1 for a in roster if a.get("alive")),
            "fallback": self.fallback.status(),
            "chain": self.chain.stats(),
            "chain_health": self.chain.health(),
            "mesh": self.mesh.stats(),
            "store_count": self.store.count(),
            "audit_len": len(self.audit),
            "audit_tail": self.audit.tail,
        }

    def pair(self, agent_id: str, manifest: Optional[Dict[str, Any]] = None,
             paired_by: str = "operator") -> Dict[str, Any]:
        return self.registry.pair(agent_id, manifest=manifest,
                                  paired_by=paired_by)

    def unpair(self, agent_id: str) -> bool:
        return self.registry.unpair(agent_id)

    def resume(self, attributed_by: str) -> None:
        if self.mode == "halted":
            return
        self.fallback.resume(resumed_by=attributed_by)
        self._last_lost_report.clear()

    def broadcast_system(self, payload: Dict[str, Any]) -> int:
        env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                       sender=self.agent_id, recipient="*",
                       topic=f"/shugunet/{self.agent_id}/system",
                       payload=payload)
        report = self.chain.send(env)
        return len(self._paired_agents_excluding(self.agent_id)) if report.ok else 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shogunet fleet host")
    parser.add_argument("--agent-id", default=DEFAULT_HOST_ID)
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=9000)
    parser.add_argument("--relay-port", type=int, default=9001)
    parser.add_argument("--dashboard-port", type=int, default=0,
                        help="Serve the fleet dashboard on this port (e.g. 9002)")
    parser.add_argument("--dashboard-bind", default="127.0.0.1",
                        help="Dashboard bind address (loopback by default)")
    parser.add_argument("--dashboard-token", default=None,
                        help="Require this token on dashboard POST endpoints")
    parser.add_argument("--pair", action="append", default=[],
                        help="Pre-pair an agent ID (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    host = ShugonetHost(agent_id=args.agent_id,
                        tcp_host=args.tcp_host,
                        tcp_port=args.tcp_port,
                        relay_port=args.relay_port,
                        dashboard_port=args.dashboard_port,
                        dashboard_bind=args.dashboard_bind,
                        dashboard_token=args.dashboard_token)
    host.start()
    for agent_id in args.pair:
        host.pair(agent_id)
        print(f"paired: {agent_id}")
    print(f"Host '{args.agent_id}' ready — TCP :{args.tcp_port}, "
          f"relay :{args.relay_port}")
    if args.dashboard_port:
        print(f"Dashboard :{args.dashboard_port}"
              f"{' (token required)' if args.dashboard_token else ''}")
    print("Press Ctrl-C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        host.stop()
        print("Stopped.")