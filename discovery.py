"""
UDP-multicast discovery for IP-class media (WiFi, WiFi-Halow, LAN sides of
5G/4G/EDGE where devices share a link).

Beacons are protocol envelopes (``announce``, compact codec) sent to a
multicast group; listeners feed the AgentRegistry's liveness and notify an
``on_peer`` callback so the fallback chain can learn where peers are
reachable. Bluetooth pairs explicitly, LoRa beacons on its own duty-cycled
radio, cellular rendezvouses at the relay hub -- each medium gets the
discovery strategy its physics allows; this module covers IP multicast.
"""

import logging
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, Optional

from protocol import (CODEC_COMPACT, Envelope, ProtocolError, decode, encode,
                      new_msg_id)
from security import sanitize_text

logger = logging.getLogger(__name__)

MCAST_GROUP = "239.77.78.79"
MCAST_PORT = 47779
BEACON_INTERVAL_S = 10.0


class UDPMulticastDiscovery:
    """Announce/heartbeat beacons over a multicast group."""

    def __init__(self, agent_id: str, registry: Optional[Any] = None,
                 group: str = MCAST_GROUP, port: int = MCAST_PORT,
                 interval_s: float = BEACON_INTERVAL_S,
                 manifest: Optional[Dict[str, Any]] = None, realm: str = "*",
                 on_peer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 audit: Optional[Any] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self.registry = registry
        self.group = str(group)
        self.port = max(1, min(65535, int(port)))
        self.interval_s = max(0.05, float(interval_s))
        self.manifest = dict(manifest or {})
        self.realm = sanitize_text(realm, 8) or "*"
        self.on_peer = on_peer
        self.audit = audit
        self._tx: Optional[socket.socket] = None
        self._rx: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._threads: list = []
        self._peers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stats = {"beacons_sent": 0, "beacons_received": 0,
                       "rejected": 0, "peers_seen": 0}

    # -- sockets -----------------------------------------------------------------

    def _open_sockets(self) -> None:
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            tx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        try:
            # Multicast beacons must not escape the local site.
            tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        except OSError:
            pass
        self._tx = tx
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        rx.bind(("", self.port))
        mreq = struct.pack("4sl", socket.inet_aton(self.group),
                           socket.INADDR_ANY)
        rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        rx.settimeout(0.25)
        self._rx = rx

    # -- beacons ---------------------------------------------------------------

    def beacon(self) -> bool:
        """Send one announce beacon now."""
        if self._tx is None:
            return False
        env = Envelope(msg_id=new_msg_id(), msg_type="announce",
                       sender=self.agent_id, recipient="*",
                       realm=self.realm,
                       payload={"manifest": self.manifest})
        try:
            self._tx.sendto(encode(env, CODEC_COMPACT),
                            (self.group, self.port))
        except OSError as exc:
            logger.warning("beacon send failed: %s", exc)
            return False
        self._stats["beacons_sent"] += 1
        return True

    def _beacon_loop(self) -> None:
        self.beacon()
        while not self._stop.wait(self.interval_s):
            self.beacon()

    def _listen_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _addr = self._rx.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_datagram(data)

    def _handle_datagram(self, data: bytes) -> None:
        self._stats["beacons_received"] += 1
        try:
            env = decode(data)
        except ProtocolError:
            self._stats["rejected"] += 1
            return
        sender = env.sender
        if (env.msg_type != "announce" or not sender
                or sender == self.agent_id):
            self._stats["rejected"] += 1
            return
        manifest = env.payload.get("manifest")
        with self._lock:
            if sender not in self._peers:
                self._stats["peers_seen"] += 1
            self._peers[sender] = {
                "last_seen": time.monotonic(),
                "manifest": manifest if isinstance(manifest, dict) else {},
                "realm": env.realm,
            }
        if self.registry is not None:
            self.registry.heartbeat(sender)
        if self.on_peer is not None:
            try:
                self.on_peer(sender, self.manifest_of(sender))
            except Exception:
                logger.warning("on_peer callback failed", exc_info=True)

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        self._open_sockets()
        self._stop.clear()
        for target, name in ((self._beacon_loop, "sgn-disc-tx"),
                             (self._listen_loop, "sgn-disc-rx")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []
        for sock in (self._tx, self._rx):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._tx = self._rx = None

    # -- roster -------------------------------------------------------------------

    def manifest_of(self, agent_id: str) -> Dict[str, Any]:
        with self._lock:
            peer = self._peers.get(str(agent_id))
        return dict(peer["manifest"]) if peer else {}

    def peers(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {agent: dict(peer) for agent, peer in self._peers.items()}

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

