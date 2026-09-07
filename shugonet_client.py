"""
Shugonet client — cross-platform CLI and programmatic API
=========================================================

A stdlib-only client for connecting to a ShugonetHost fleet.  Works on
macOS, Linux, and Windows (including Termux on Android).  The client wraps
``ShugonetAgentRuntime`` and exposes the standard contract
(``send`` / ``query`` / ``sync`` / ``list_agents`` / ``status``) expected
by ``shugocore_adapter.ShugonetExecutionHandler``.

Two entry points
----------------

1. **Programmatic** — instantiate ``ShugonetClient``, call ``connect()``,
   use the methods, then ``disconnect()``.

2. **CLI** — ``shugonet-client`` (installed by ``pip``) with subcommands:

   ::

       shugonet-client run              # foreground daemon (connects to host)
       shugonet-client status           # one-shot status query
       shugonet-client send <peer> <topic> <json>   # one-shot send
       shugonet-client query <text>     # one-shot memory query
       shugonet-client sync [peer]      # one-shot digest exchange

Configuration (all subcommands)
-------------------------------

Reads from environment variables (or CLI flags):

    SHUGONET_AGENT_ID     agent identifier (default: hostname)
    SHUGONET_HOST         host address   (default: 127.0.0.1)
    SHUGONET_TCP_PORT     TCP port       (default: 9000)
    SHUGONET_RELAY_URL    relay hub URL  (optional)
    SHUGONET_REALM        sim | phys     (default: phys)
    SHUGONET_LOG_LEVEL    debug | info | warning (default: info)
"""

import json
import logging
import os
import platform
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import version
from shugonet_runtime import ShugonetAgentRuntime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_ENV_PREFIX = "SHUGONET_"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(f"{_ENV_PREFIX}{key}", default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except (TypeError, ValueError):
        return default
# ---------------------------------------------------------------------------
# ShugonetClient — programmatic API
# ---------------------------------------------------------------------------


class ShugonetClient:
    """High-level client that wraps ``ShugonetAgentRuntime``.

    Typical usage::

        client = ShugonetClient(agent_id="robot-1")
        client.connect()
        client.send("robot-2", "/shugunet/robot-1/status", {"battery": 87})
        print(client.status())
        client.disconnect()

    All configuration can be supplied as constructor arguments or via
    ``SHUGONET_*`` environment variables.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        host: str = "127.0.0.1",
        tcp_port: int = 9000,
        relay_url: Optional[str] = None,
        realm: str = "phys",
        log_level: str = "info",
    ):
        # Resolve agent_id: explicit arg > env var > hostname
        self.agent_id = (
            agent_id
            or _env("AGENT_ID")
            or platform.node()
            or "shugonet-client"
        )
        self.host = host or _env("HOST", "127.0.0.1")
        self.tcp_port = tcp_port or _env_int("TCP_PORT", 9000)
        self.relay_url = relay_url or _env("RELAY_URL") or None
        self.realm = realm or _env("REALM", "phys")
        self.log_level = (log_level or _env("LOG_LEVEL", "info")).upper()

        self._runtime: Optional[ShugonetAgentRuntime] = None
        self._connected = False

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the ShugonetHost.  Returns True on success."""
        if self._connected:
            return True
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        self._runtime = ShugonetAgentRuntime(
            agent_id=self.agent_id,
            host_tcp_host=self.host,
            host_tcp_port=self.tcp_port,
            host_relay_url=self.relay_url,
            realm=self.realm,
        )
        try:
            self._runtime.connect_to_host()
            # Give the handshake a moment to complete.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if self._runtime.tcp and self._runtime.tcp.has_active_connection():
                    self._connected = True
                    break
                time.sleep(0.05)
            if not self._connected:
                logger.warning("connect: no active TCP connection after handshake")
                return False
            logger.info(
                "connected as '%s' to %s:%s",
                self.agent_id, self.host, self.tcp_port,
            )
            return True
        except Exception as exc:
            logger.error("connect failed: %s", exc)
            self._runtime = None
            return False

    def disconnect(self) -> None:
        """Disconnect from the host and stop background threads."""
        if self._runtime is not None:
            try:
                self._runtime.stop()
            except Exception as exc:
                logger.warning("disconnect: %s", exc)
        self._runtime = None
        self._connected = False

    # -- API surface (send / query / sync / list_agents / status) ------------

    def send(
        self, peer: str, topic: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send an addressed message to a peer via the host."""
        if self._runtime is None:
            return {"status": "refused", "reason": "not connected"}
        return self._runtime.send(peer, topic, payload)

    def query(
        self, text: str, peers: Optional[List[str]] = None, top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """Fan-out a memory query to online peers."""
        if self._runtime is None:
            return []
        return self._runtime.query(text, peers=peers, top_k=top_k)

    def sync(self, peer: Optional[str] = None) -> Dict[str, Any]:
        """Trigger a fact-digest sync with a peer (or broadcast)."""
        if self._runtime is None:
            return {"status": "refused", "reason": "not connected"}
        return self._runtime.sync(peer=peer)

    def list_agents(self) -> List[Dict[str, Any]]:
        """Return the roster of paired agents from the host."""
        if self._runtime is None:
            return []
        return self._runtime.list_agents()

    def status(self) -> Dict[str, Any]:
        """Return the client's current status snapshot."""
        if self._runtime is None:
            return {"agent_id": self.agent_id, "connected": False}
        base = self._runtime.status()
        base["connected"] = self._connected
        base["host"] = self.host
        base["tcp_port"] = self.tcp_port
        return base

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block the calling thread (Ctrl-C to stop)."""
        try:
            threading.Event().wait(timeout)
        except KeyboardInterrupt:
            self.disconnect()
            return

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli() -> Dict[str, Any]:
    """Minimal CLI parser (stdlib-only)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="shugonet-client",
        description="Shugonet cross-platform client",
    )
    parser.add_argument(
        "--agent-id", default=None,
        help="Agent identifier (default: hostname / SHUGONET_AGENT_ID)",
    )
    parser.add_argument(
        "--host", default=None,
        help="ShugonetHost address (default: 127.0.0.1 / SHUGONET_HOST)",
    )
    parser.add_argument(
        "--tcp-port", type=int, default=None,
        help="ShugonetHost TCP port (default: 9000 / SHUGONET_TCP_PORT)",
    )
    parser.add_argument(
        "--relay-url", default=None,
        help="Relay hub URL (optional / SHUGONET_RELAY_URL)",
    )
    parser.add_argument(
        "--realm", default=None, choices=("sim", "phys"),
        help="Agent realm (default: phys / SHUGONET_REALM)",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=("debug", "info", "warning", "error"),
        help="Log level (default: info / SHUGONET_LOG_LEVEL)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run — foreground daemon
    run_parser = subparsers.add_parser(
        "run", help="Connect to the host and stay connected (foreground daemon)"
    )
    run_parser.add_argument(
        "--timeout", type=float, default=None,
        help="Exit after N seconds (default: run forever)",
    )

    # status — one-shot
    subparsers.add_parser("status", help="One-shot status query")

    # send — one-shot
    send_parser = subparsers.add_parser("send", help="One-shot send to a peer")
    send_parser.add_argument("peer", help="Recipient agent ID")
    send_parser.add_argument("topic", help="Topic string")
    send_parser.add_argument(
        "payload", help="JSON payload (stringified object)",
    )

    # query — one-shot
    query_parser = subparsers.add_parser("query", help="One-shot memory query")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument(
        "--top-k", type=int, default=8, help="Max results (default: 8)",
    )
    query_parser.add_argument(
        "--peers", nargs="*", default=None,
        help="Peer agent IDs to query",
    )

    # sync — one-shot
    sync_parser = subparsers.add_parser("sync", help="One-shot digest sync")
    sync_parser.add_argument(
        "peer", nargs="?", default=None,
        help="Peer agent ID (default: broadcast)",
    )

# observe — one-shot spatial observation
    obs_parser = subparsers.add_parser(
        "observe", help="Publish a spatial observation"
    )
    obs_parser.add_argument("entity_id", help="Entity identifier")
    obs_parser.add_argument("x", type=float, help="X coordinate")
    obs_parser.add_argument("y", type=float, help="Y coordinate")
    obs_parser.add_argument("z", type=float, help="Z coordinate")
    obs_parser.add_argument("--confidence", type=float, default=1.0,
                            help="Observation confidence (0-1)")
    obs_parser.add_argument("--label", default="",
                            help="Human-readable label")
    obs_parser.add_argument("--frame-id", default="world",
                            help="Coordinate frame")

    # map — consolidated fleet spatial view
    subparsers.add_parser("map", help="Show the fleet's spatial map")

    # locate — find an agent's position
    locate_parser = subparsers.add_parser(
        "locate", help="Show one agent's position"
    )
    locate_parser.add_argument("agent_id", help="Agent to locate")

    # nearby — spatial query near a point
    nearby_parser = subparsers.add_parser(
        "nearby", help="Query observations near a point"
    )
    nearby_parser.add_argument("x", type=float, help="X coordinate")
    nearby_parser.add_argument("y", type=float, help="Y coordinate")
    nearby_parser.add_argument("z", type=float, help="Z coordinate")
    nearby_parser.add_argument("radius", type=float,
                               help="Search radius in metres")
    return vars(parser.parse_args())


def _build_client(args: Dict[str, Any]) -> ShugonetClient:
    return ShugonetClient(
        agent_id=args.get("agent_id"),
        host=args.get("host") or "127.0.0.1",
        tcp_port=args.get("tcp_port") or 9000,
        relay_url=args.get("relay_url"),
        realm=args.get("realm") or "phys",
        log_level=args.get("log_level") or "info",
    )


def cli_main() -> None:
    """CLI entry point (setuptools console_scripts)."""
    args = _parse_cli()
    command = args.pop("command")
    client = _build_client(args)

    if command == "run":
        _cmd_run(client, args)
    elif command == "status":
        _cmd_status(client)
    elif command == "send":
        _cmd_send(client, args)
    elif command == "query":
        _cmd_query(client, args)
    elif command == "sync":
        _cmd_sync(client, args)
    elif command == "observe":
        _cmd_observe(client, args)
    elif command == "map":
        _cmd_map(client)
    elif command == "locate":
        _cmd_locate(client, args)
    elif command == "nearby":
        _cmd_nearby(client, args)


# -- subcommand implementations ------------------------------------------------


def _cmd_run(client: ShugonetClient, args: Dict[str, Any]) -> None:
    """Foreground daemon — connect and stay connected."""
    timeout = args.get("timeout")
    if not client.connect():
        logger.error("failed to connect; exiting")
        sys.exit(1)
    print(f"Shugonet client '{client.agent_id}' connected — Ctrl-C to stop.")
    try:
        client.wait(timeout)
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        print("disconnected.")


def _cmd_status(client: ShugonetClient) -> None:
    """One-shot status."""
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        st = client.status()
        print(json.dumps(st, indent=2, sort_keys=True))
    finally:
        client.disconnect()


def _cmd_send(client: ShugonetClient, args: Dict[str, Any]) -> None:
    """One-shot send."""
    peer = args["peer"]
    topic = args["topic"]
    try:
        payload = json.loads(args["payload"])
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("invalid JSON payload: %s", exc)
        sys.exit(1)
    if not isinstance(payload, dict):
        logger.error("payload must be a JSON object")
        sys.exit(1)
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        result = client.send(peer, topic, payload)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        client.disconnect()


def _cmd_query(client: ShugonetClient, args: Dict[str, Any]) -> None:
    """One-shot memory query."""
    text = args["text"]
    top_k = args.get("top_k", 8)
    peers = args.get("peers")
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        results = client.query(text, peers=peers, top_k=top_k)
        print(json.dumps(results, indent=2, sort_keys=True))
    finally:
        client.disconnect()


def _cmd_sync(client: ShugonetClient, args: Dict[str, Any]) -> None:
    """One-shot digest sync."""
    peer = args.get("peer")
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        result = client.sync(peer=peer)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        client.disconnect()


# -- spatial subcommands ---------------------------------------------------


def _cmd_observe(client: ShugonetClient, args: Dict[str, Any]) -> None:
    """One-shot spatial observation."""
    entity_id = args["entity_id"]
    x = args["x"]
    y = args["y"]
    z = args["z"]
    confidence = args.get("confidence", 1.0)
    label = args.get("label", "")
    frame_id = args.get("frame_id", "world")
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        result = client._runtime.publish_observation(
            entity_id, x, y, z,
            confidence=confidence, label=label, frame_id=frame_id)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        client.disconnect()


def _cmd_map(client: ShugonetClient) -> None:
    """Show the fleet's consolidated spatial map."""
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        fleet_map = client._runtime.get_fleet_map()
        print(json.dumps(fleet_map, indent=2, sort_keys=True))
    finally:
        client.disconnect()


def _cmd_locate(client: ShugonetClient, args: Dict[str, Any]) -> None:
    """Show one agent's position."""
    agent_id = args["agent_id"]
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        pos = client._runtime.spatial_sync.get_agent_position(agent_id)
        if pos:
            print(json.dumps(pos.to_dict(), indent=2, sort_keys=True))
        else:
            print(json.dumps({"agent_id": agent_id, "position": None},
                             indent=2, sort_keys=True))
    finally:
        client.disconnect()


def _cmd_nearby(client: ShugonetClient, args: Dict[str, Any]) -> None:
    """Query spatial observations near a point."""
    x = args["x"]
    y = args["y"]
    z = args["z"]
    radius = args["radius"]
    if not client.connect():
        logger.error("failed to connect")
        sys.exit(1)
    try:
        results = client._runtime.query_nearby(x, y, z, radius)
        if results:
            print(json.dumps(results, indent=2, sort_keys=True))
        else:
            print(json.dumps([], indent=2, sort_keys=True))
    finally:
        client.disconnect()
# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli_main()

    def disconnect(self) -> None:
        """Disconnect from the host and stop background threads."""
        if self._runtime is not None:
            try:
                self._runtime.stop()
            except Exception as exc:
                logger.warning("disconnect: %s", exc)
        self._runtime = None
        self._connected = False