"""
Shogunet transport layer
========================

``BaseTransport`` is the single interface every Shogunet transport implements,
mirroring ShugoCore's ``BaseROS2Interface`` pattern: capability probe
(``is_available``), lifecycle (``start`` / ``stop``), bounded inbound dispatch
(``poll`` drives user callbacks -- no transport thread ever enters user code
directly), and ``send_frame`` for serialized protocol frames.

``LinkProfile`` captures the physical reality of each network named in the
project description -- 5G, 4G, EDGE, LoRa, WiFi-Halow, WiFi, Bluetooth -- with
conservative numbers for bandwidth, latency, per-chunk payload caps and duty
cycle. The fallback chain and the link simulator both read these profiles;
they are policy inputs, not channel measurements.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

from protocol import MAX_ENVELOPE_BYTES, ProtocolError, peek_header
from security import sanitize_text

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class LinkProfile:
    """Physical characteristics of one target network."""

    name: str
    kbps: float                  # sustained throughput (0 = unthrottled)
    latency_s: float             # typical one-way latency
    max_payload_bytes: int       # per-chunk cap for datagram media (MTU-like)
    duty_cycle: Optional[float]  # fraction of airtime allowed (None = unlimited)
    supports_ip: bool            # False => rendezvous needs discovery/hub/pairing
    segmented: bool = False      # medium tolerates app-layer segmentation


LINK_PROFILES: Dict[str, LinkProfile] = {
    "loopback":   LinkProfile("loopback", 0, 0.0, MAX_ENVELOPE_BYTES, None, True),
    "wifi":       LinkProfile("wifi", 50_000, 0.003, 1400, None, True),
    "5g":         LinkProfile("5g", 30_000, 0.03, 1400, None, True),
    "4g":         LinkProfile("4g", 8_000, 0.08, 1400, None, True),
    "edge":       LinkProfile("edge", 100, 0.9, 1400, None, True),
    "wifi_halow": LinkProfile("wifi_halow", 120, 0.15, 1400, None, True),
    "bluetooth":  LinkProfile("bluetooth", 250, 0.05, 990, None, False),
    "ble":        LinkProfile("ble", 100, 0.1, 247, None, False, True),
    "lora":       LinkProfile("lora", 5.0, 2.0, 220, 0.01, False),
}


def profile_for(name: str) -> LinkProfile:
    """Resolve a profile by name (loopback default)."""
    key = str(name or "loopback").strip().lower()
    if key not in LINK_PROFILES:
        raise ProtocolError(f"unknown link profile '{name}'")
    return LINK_PROFILES[key]


class BaseTransport:
    """Interface for all Shogunet transports (real and simulated)."""

    name = "base"
    profile: LinkProfile = LINK_PROFILES["loopback"]

    def is_available(self) -> bool:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def send_frame(self, peer: Optional[str], frame: bytes) -> bool:
        """Send one serialized frame. ``peer`` is an agent_id or "*" broadcast."""
        raise NotImplementedError

    def subscribe(self, callback: Callable[[str, bytes], None]) -> None:
        """Register ``(sender_agent_id, frame)`` callback, fired from poll()."""
        raise NotImplementedError

    def poll(self, timeout: float = 0.1) -> None:
        """Drain inbound frames and dispatch to subscribers (spin-once)."""
        raise NotImplementedError

    def stats(self) -> Dict[str, object]:
        raise NotImplementedError

    # -- shared policy helpers ---------------------------------------------

    broadcast_supported = True
    """Whether ``send_frame`` accepts ``recipient == "*"``. Transports that
    are strictly addressed (the relay hub has no fleet-wide mailbox) declare
    False so the fallback chain skips them for broadcasts instead of counting
    a healthy transport as failing."""

    def accepts_class(self, priority_class: int) -> bool:
        """Priority-class eligibility for the fallback chain.

        Datagram media without segmentation can only hold small envelopes, so
        they carry control (P0) and memory-delta (P1) classes only; segmented
        or IP-capable media carry everything.
        """
        if self.profile.supports_ip or self.profile.segmented:
            return True
        return self.profile.max_payload_bytes >= 1024 or priority_class <= 1

    def chunk_size(self) -> int:
        """Largest single wire chunk this medium accepts."""
        return min(self.profile.max_payload_bytes, MAX_ENVELOPE_BYTES)


class LoopbackBus:
    """A virtual in-process ether shared by every LoopbackTransport that
    joins it. Delivery is immediate (no impairment); use link_simulator for
    constrained-link behaviour."""

    def __init__(self, name: str = "default"):
        self.name = str(name)
        self._members: List["LoopbackTransport"] = []
        self._lock = threading.Lock()

    def join(self, transport: "LoopbackTransport") -> None:
        with self._lock:
            if transport not in self._members:
                self._members.append(transport)

    def leave(self, transport: "LoopbackTransport") -> None:
        with self._lock:
            if transport in self._members:
                self._members.remove(transport)

    def broadcast(self, sender: "LoopbackTransport", peer: Optional[str],
                  frame: bytes) -> int:
        """Deliver to every member except the sender; returns delivered count."""
        with self._lock:
            targets = list(self._members)
        delivered = 0
        for member in targets:
            if member is sender:
                continue
            if peer and peer != "*":
                if member.agent_id != sanitize_text(peer, 48):
                    continue
            if member._deliver(sender.agent_id, frame):
                delivered += 1
        return delivered


class LoopbackTransport(BaseTransport):
    """In-process transport for same-host agents and the test suite.

    Frames are queued bounded and dispatched to subscribers on ``poll()``,
    exactly like every other transport -- code written against loopback runs
    unchanged over TCP, relay, LoRa or Bluetooth. Transports only talk when
    they explicitly share a ``LoopbackBus``.
    """

    name = "loopback"

    def __init__(self, agent_id: str, bus: Optional[LoopbackBus] = None,
                 max_queue: int = 256):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self.bus = bus or LoopbackBus(f"bus-{self.agent_id}")
        self.profile = LINK_PROFILES["loopback"]
        self._inbox: Deque[Tuple[str, bytes]] = deque(maxlen=max(1, int(max_queue)))
        self._callbacks: List[Callable[[str, bytes], None]] = []
        self._lock = threading.Lock()
        self._running = False
        self._stats = {"sent": 0, "send_failed": 0, "received": 0,
                       "dropped": 0}
        self.bus.join(self)

    def is_available(self) -> bool:
        return True

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.bus.leave(self)

    def send_frame(self, peer: Optional[str], frame: bytes) -> bool:
        frame = bytes(frame)
        if len(frame) > MAX_ENVELOPE_BYTES:
            with self._lock:
                self._stats["send_failed"] += 1
            return False
        try:
            peek_header(frame)   # never inject invalid frames, even locally
        except ProtocolError:
            with self._lock:
                self._stats["send_failed"] += 1
            return False
        delivered = self.bus.broadcast(self, peer, frame)
        if delivered == 0 and peer not in (None, "", "*"):
            with self._lock:
                self._stats["send_failed"] += 1
            return False
        with self._lock:
            self._stats["sent"] += 1
        return True

    def subscribe(self, callback: Callable[[str, bytes], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def _deliver(self, sender: str, frame: bytes) -> bool:
        with self._lock:
            if len(self._inbox) == self._inbox.maxlen:
                self._stats["dropped"] += 1
                self._inbox.popleft()   # bounded: drop oldest, newest wins
            self._inbox.append((sender, bytes(frame)))
        return True

    def poll(self, timeout: float = 0.1) -> None:
        while True:
            with self._lock:
                if not self._inbox:
                    break
                sender, frame = self._inbox.popleft()
                self._stats["received"] += 1
            for callback in list(self._callbacks):
                try:
                    callback(sender, frame)
                except Exception:   # subscriber bugs never kill the poll loop
                    logger.warning("loopback callback failed", exc_info=True)

    def stats(self) -> Dict[str, object]:
        return dict(self._stats)



# [SGN:CONT]
