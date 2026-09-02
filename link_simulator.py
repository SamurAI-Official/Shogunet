"""
Shogunet in-process link simulator
==================================

A dependency-free "netem": ``SimulatedEther`` wraps a virtual broadcast ether
and impairs it according to a LinkProfile -- bandwidth-derived serialization
delay, latency, jitter, random loss, MTU rejection and duty-cycle airtime
budgeting. ``SimulatedTransport`` endpoints join an ether and behave exactly
like any other ``BaseTransport``, so the whole test suite exercises 5G, 4G,
EDGE, LoRa, WiFi-Halow, WiFi and Bluetooth behaviour without hardware.

The simulator is a policy test tool: its numbers are deliberately conservative
approximations (see ``transports.LINK_PROFILES``), not channel measurements.
Duty-cycle budgeting tracks airtime over a rolling one-hour window, matching
the sub-GHz ISM band rules that make LoRa links so sparse.
"""

import logging
import random
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from protocol import ProtocolError
from security import sanitize_text
from transports import LINK_PROFILES, BaseTransport, LinkProfile

logger = logging.getLogger(__name__)


_WINDOW_S = 3600.0
_SCHEDULED_CAP = 4096


class SimulatedEther:
    """Impaired virtual ether shared by SimulatedTransport members."""

    def __init__(self, profile, seed: Optional[int] = None,
                 loss_rate: float = 0.0, jitter_s: float = 0.0,
                 scheduled_cap: int = _SCHEDULED_CAP):
        self.profile = (profile if isinstance(profile, LinkProfile)
                        else LINK_PROFILES[profile])
        self.loss_rate = max(0.0, min(1.0, float(loss_rate)))
        self.jitter_s = max(0.0, float(jitter_s))
        self._rng = random.Random(seed)
        self._members: List["SimulatedTransport"] = []
        self._scheduled: Deque[Tuple[float, str, str, bytes]] = deque(
            maxlen=max(1, int(scheduled_cap)))
        self._airtime: Deque[Tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {"submitted": 0, "delivered": 0, "dropped_loss": 0,
                       "dropped_mtu": 0, "dropped_duty": 0,
                       "dropped_overflow": 0, "dropped_no_member": 0}

    # -- membership ------------------------------------------------------------

    def join(self, member: "SimulatedTransport") -> None:
        with self._lock:
            if member not in self._members:
                self._members.append(member)

    def leave(self, member: "SimulatedTransport") -> None:
        with self._lock:
            if member in self._members:
                self._members.remove(member)

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        with self._wake:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run, name="sgn-ether",
                                        daemon=True)
        self._thread.start()

    def close(self) -> None:
        with self._wake:
            self._running = False
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while True:
            with self._wake:
                if not self._running:
                    return
                now = time.monotonic()
                due = self._scheduled[0][0] if self._scheduled else None
                if due is None or due > now:
                    self._wake.wait(timeout=(0.05 if due is None
                                             else max(0.005, due - now)))
                    continue
                _, sender, recipient, frame = self._scheduled.popleft()
            self._dispatch(sender, recipient, frame)

    def _dispatch(self, sender: str, recipient: str, frame: bytes) -> None:
        with self._lock:
            targets = [m for m in self._members
                       if m.agent_id != sender
                       and (recipient in ("*", "") or m.agent_id == recipient)]
            if not targets:
                self._stats["dropped_no_member"] += 1
            for member in targets:
                if member._deliver(sender, frame):
                    self._stats["delivered"] += 1

    # -- impairment ------------------------------------------------------------

    def _airtime_s(self, frame_len: int) -> float:
        """On-air time for one frame at this profile's sustained rate."""
        if self.profile.kbps <= 0:
            return 0.0
        return (frame_len * 8.0) / (self.profile.kbps * 1000.0)

    def _duty_allows(self, start: float, airtime: float) -> bool:
        """Rolling-window airtime budget: within any 1-hour window the medium
        may transmit at most ``duty_cycle * 3600`` seconds."""
        if self.profile.duty_cycle is None:
            return True
        budget = self.profile.duty_cycle * _WINDOW_S
        window_end = start + airtime
        window_start = window_end - _WINDOW_S
        used = 0.0
        for t_start, t_end in self._airtime:
            overlap = min(t_end, window_end) - max(t_start, window_start)
            if overlap > 0:
                used += overlap
        return used + airtime <= budget + 1e-9

    def submit(self, sender: str, recipient: Optional[str], frame: bytes) -> bool:
        """Impair and schedule one frame; False when the medium refuses it
        (MTU, duty-cycle exhaustion, loss, or a full schedule)."""
        frame = bytes(frame)
        with self._lock:
            self._stats["submitted"] += 1
            if len(frame) > self.profile.max_payload_bytes:
                self._stats["dropped_mtu"] += 1
                return False
            now = time.monotonic()
            airtime = self._airtime_s(len(frame))
            if not self._duty_allows(now, airtime):
                self._stats["dropped_duty"] += 1
                return False
            if airtime > 0:
                self._airtime.append((now, now + airtime))
                while self._airtime and self._airtime[0][1] < now - _WINDOW_S:
                    self._airtime.popleft()
            if self.loss_rate > 0 and self._rng.random() < self.loss_rate:
                self._stats["dropped_loss"] += 1
                return False
            delay = self.profile.latency_s + airtime
            if self.jitter_s:
                delay += self._rng.uniform(0.0, self.jitter_s)
            if len(self._scheduled) == self._scheduled.maxlen:
                self._stats["dropped_overflow"] += 1
                return False
            self._scheduled.append((now + delay, sender, recipient or "*", frame))
            self._wake.notify_all()
        return True

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every scheduled frame has been dispatched (test aid)."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            with self._lock:
                if not self._scheduled:
                    return True
            time.sleep(0.01)
        return False

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)


class SimulatedTransport(BaseTransport):
    """A transport endpoint attached to a SimulatedEther.

    Identical surface to every other transport; impairment happens in the
    ether, so per-endpoint code has no idea it is being throttled, lossy or
    duty-cycled -- exactly like real radio hardware.
    """

    name = "simulated"

    def __init__(self, agent_id: str, ether: SimulatedEther,
                 max_queue: int = 256):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self.ether = ether
        self.profile = ether.profile
        self._inbox: Deque[Tuple[str, bytes]] = deque(maxlen=max(1, int(max_queue)))
        self._callbacks: List[Callable[[str, bytes], None]] = []
        self._lock = threading.Lock()
        self._running = False
        self._stats = {"sent": 0, "send_failed": 0, "received": 0, "dropped": 0}
        ether.join(self)

    def is_available(self) -> bool:
        return True

    def start(self) -> None:
        self._running = True
        self.ether.start()

    def stop(self) -> None:
        self._running = False
        self.ether.leave(self)

    def send_frame(self, peer: Optional[str], frame: bytes) -> bool:
        ok = self.ether.submit(self.agent_id, peer, bytes(frame))
        with self._lock:
            self._stats["sent" if ok else "send_failed"] += 1
        return ok

    def subscribe(self, callback: Callable[[str, bytes], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def _deliver(self, sender: str, frame: bytes) -> bool:
        with self._lock:
            if len(self._inbox) == self._inbox.maxlen:
                self._stats["dropped"] += 1
                self._inbox.popleft()
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
                    logger.warning("simulated callback failed", exc_info=True)

    def stats(self) -> Dict[str, object]:
        return dict(self._stats)



def make_bus(agent_ids: List[str], profile, seed: Optional[int] = None,
             loss_rate: float = 0.0, jitter_s: float = 0.0):
    """Convenience: one impaired ether with a transport endpoint per agent."""
    ether = SimulatedEther(profile, seed=seed, loss_rate=loss_rate,
                           jitter_s=jitter_s)
    transports = [SimulatedTransport(agent_id, ether) for agent_id in agent_ids]
    ether.start()
    return ether, transports


