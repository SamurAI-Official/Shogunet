"""
Shogunet transport fallback chain
=================================

The deterministic heart of "network over different networks in fallback
methods": an ordered, health-ranked chain of transports behind one API.

Per message:
1. Filter transports by priority-class eligibility (``accepts_class``):
   constrained datagram media (LoRa) never see P2/P3 envelopes.
2. Order candidates by health: closed breakers first, then EWMA latency
   ascending (open breakers sink to half-open probe duty).
3. Walk the chain until one transport accepts the frame. Encode per
   transport: constrained media get the compact codec, broadband gets JSON.
4. Frames larger than the medium's chunk cap are segmented when the profile
   allows it (BLE), otherwise the transport is skipped.

Circuit breakers mirror ShugoCore's security breakers: ``threshold``
consecutive failures open a transport for ``cooldown_s``; after cooldown a
single half-open probe is allowed. Success resets the breaker and records
latency into the health EWMA.

QoS: ``best_effort`` returns after the chain walk; ``at_least_once``
persists the frame in an OutboxStore before sending and keeps it there
until a matching ``ack`` envelope (payload ``msg_id``) arrives through
``poll()`` -- a crash mid-send never loses the message.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from protocol import (CODEC_COMPACT, CODEC_JSON, Envelope, ProtocolError,
                      SegmentAssembler, decode, encode, segment_frame)
from security import sanitize_text
from store_forward import InboxDedup, OutboxStore
from transports import BaseTransport

logger = logging.getLogger(__name__)


@dataclass
class SendReport:
    """Outcome of one chain walk."""

    ok: bool
    via: Optional[str] = None
    attempts: int = 0
    skipped: List[str] = field(default_factory=list)
    codec: Optional[int] = None
    segmented: bool = False


class _Circuit:
    """Per-transport breaker state plus a health EWMA."""

    __slots__ = ("failures", "open_until", "latency_ewma")

    def __init__(self) -> None:
        self.failures = 0
        self.open_until = 0.0
        self.latency_ewma: Optional[float] = None


class TransportChain:
    """Fallback chain over any set of transports (real or simulated)."""

    def __init__(self, agent_id: str, transports: List[BaseTransport],
                 outbox: Optional[OutboxStore] = None,
                 breaker_threshold: int = 3, breaker_cooldown_s: float = 5.0,
                 dedup_capacity: int = 4096, audit: Optional[Any] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self._transports = list(transports)
        self.outbox = outbox
        self.breaker_threshold = max(1, int(breaker_threshold))
        self.breaker_cooldown_s = max(0.1, float(breaker_cooldown_s))
        self.audit = audit
        self._circuits: Dict[str, _Circuit] = {t.name: _Circuit()
                                               for t in self._transports}
        self._dedup = InboxDedup(dedup_capacity)
        self._assembler = SegmentAssembler()
        self._handlers: List[Callable[[Envelope, str], None]] = []
        self._routers: List[Callable[[Envelope, str], bool]] = []
        self._lock = threading.RLock()
        self._stats = {"sent": 0, "send_failed": 0, "received": 0,
                       "duplicates": 0, "decoded": 0, "bad_frames": 0,
                       "acked": 0}
        for transport in self._transports:
            transport.subscribe(self._on_frame)

    # -- send path ------------------------------------------------------------

    def _ranked(self, priority_class: int) -> List[BaseTransport]:
        """Class-eligible transports, healthy first, then EWMA latency."""
        now = time.monotonic()

        def rank(transport: BaseTransport):
            circuit = self._circuits[transport.name]
            is_open = circuit.open_until > now
            latency = (circuit.latency_ewma
                       if circuit.latency_ewma is not None else 1e12)
            return (1 if is_open else 0, latency)

        return sorted((t for t in self._transports
                       if t.accepts_class(priority_class)), key=rank)

    def send(self, envelope: Envelope, qos: str = "best_effort") -> SendReport:
        """Walk the chain until some transport carries the envelope."""
        if not isinstance(envelope, Envelope):
            raise ProtocolError("send expects an Envelope")
        report = SendReport(ok=False)
        if qos == "at_least_once" and self.outbox is not None:
            # Persist first: a crash mid-send must not lose the message.
            self.outbox.enqueue(envelope.msg_id, envelope.recipient,
                                encode(envelope, CODEC_JSON),
                                {"msg_type": envelope.msg_type})
        for transport in self._ranked(envelope.priority()):
            circuit = self._circuits[transport.name]
            if circuit.open_until > time.monotonic():
                report.skipped.append(transport.name)
                continue
            if (envelope.recipient == "*"
                    and not getattr(transport, "broadcast_supported", True)):
                # Addressing-mode ineligibility is not a transport failure:
                # skipping keeps the breaker closed for a healthy relay.
                report.skipped.append(transport.name)
                continue
            codec = (CODEC_COMPACT if transport.chunk_size() <= 1024
                     else CODEC_JSON)
            try:
                frame = encode(envelope, codec)
            except ProtocolError as exc:
                logger.warning("encode failed on %s: %s", transport.name, exc)
                return report
            chunk = transport.chunk_size()
            if len(frame) > chunk and not transport.profile.segmented:
                report.skipped.append(transport.name)
                continue
            frames = (segment_frame(frame, chunk)
                      if len(frame) > chunk else [frame])
            started = time.monotonic()
            sent_all = True
            for piece in frames:
                if not transport.send_frame(envelope.recipient, piece):
                    sent_all = False
                    break
            latency = time.monotonic() - started
            report.attempts += 1
            if sent_all:
                report.ok = True
                report.via = transport.name
                report.codec = codec
                report.segmented = len(frames) > 1
                with self._lock:
                    circuit.failures = 0
                    circuit.latency_ewma = (
                        latency if circuit.latency_ewma is None
                        else 0.8 * circuit.latency_ewma + 0.2 * latency)
                    self._stats["sent"] += 1
                return report
            with self._lock:
                circuit.failures += 1
                circuit.latency_ewma = (circuit.latency_ewma or 0.5) + 0.5
                if circuit.failures >= self.breaker_threshold:
                    circuit.open_until = (time.monotonic()
                                          + self.breaker_cooldown_s)
                    self._audit("transport_breaker_open",
                                {"transport": transport.name})
            logger.warning("transport %s failed to send", transport.name)
        self._bump("send_failed")
        return report

    def send_raw(self, recipient: str, frame: bytes) -> SendReport:
        """Forward a pre-encoded frame to a specific recipient.

        Used by the host to route inbound mail without re-encoding. Walks
        the chain and sends on the first transport that accepts the frame.
        """
        report = SendReport(ok=False)
        now = time.monotonic()
        candidates = sorted(
            self._transports,
            key=lambda t: (0 if self._circuits[t.name].open_until <= now else 1,
                           self._circuits[t.name].latency_ewma or float("inf")))
        for transport in candidates:
            circuit = self._circuits[transport.name]
            if circuit.open_until > now:
                report.skipped.append(transport.name)
                continue
            if not transport.accepts_class(0):
                report.skipped.append(transport.name)
                continue
            try:
                if transport.send_frame(recipient, frame):
                    report.ok = True
                    report.via = transport.name
                    report.attempts += 1
                    circuit.failures = 0
                    self._bump("sent")
                    return report
                circuit.failures += 1
                report.attempts += 1
                if circuit.failures >= self.breaker_threshold:
                    circuit.open_until = now + self.breaker_cooldown_s
                    self._bump("breaker_opened")
                    self._audit("chain_transport_exhausted", {
                        "transport": transport.name,
                        "consecutive_failures": circuit.failures})
            except Exception:
                circuit.failures += 1
                report.attempts += 1
                logger.warning("send_raw on %s failed", transport.name,
                               exc_info=True)
            report.skipped.append(transport.name)
        self._bump("send_failed")
        return report

    # -- receive path -----------------------------------------------------------

    def register_router(self, router: Callable[[Envelope, str], bool]) -> None:
        """Register a routing hook that fires BEFORE the recipient guard.

        A router receives every decoded envelope (including those addressed
        to other agents) and can forward them. Returning True stops further
        routing. The host uses this to forward addressed mail and fan
        broadcasts -- without it, the recipient guard at _on_frame would
        drop all frames not addressed to this chain's agent before any
        handler sees them.
        """
        self._routers.append(router)

    def subscribe(self, handler: Callable[[Envelope, str], None]) -> None:
        """Register ``handler(envelope, transport_name)`` for inbound mail."""
        self._handlers.append(handler)

    def poll(self, timeout: float = 0.1) -> None:
        """Drain every transport; handlers fire during each transport's poll."""
        for transport in self._transports:
            transport.poll(timeout)

    def _on_frame(self, sender_peer: str, frame: bytes) -> None:
        try:
            assembled = self._assembler.add(frame)
        except ProtocolError:
            self._bump("bad_frames")
            return
        if assembled is None:
            return
        try:
            env = decode(assembled)
        except ProtocolError:
            self._bump("bad_frames")
            return
        sender = env.sender or sender_peer
        if self._dedup.is_duplicate(sender, env.msg_id):
            self._bump("duplicates")
            return
        self._bump("decoded")
        if env.msg_type == "ack" and self.outbox is not None:
            acked = env.payload.get("msg_id")
            if isinstance(acked, int) and self.outbox.mark_done(acked):
                self._bump("acked")
        # Routing hook: fires BEFORE the recipient guard so the host can
        # forward addressed mail and fan broadcasts. Routers see every
        # envelope, including those addressed to other agents.
        for router in list(self._routers):
            try:
                if router(env, sender_peer):
                    return  # router consumed the frame
            except Exception:
                logger.warning("chain router failed", exc_info=True)
        # Cross-talk guard: a chain processes only mail addressed to its own
        # agent (or the fleet). Frames intended for another agent that land on
        # this transport anyway -- TCP point-to-point links, the relay path,
        # broadcast media -- must never reach handlers. Without this, a
        # ``memory_fact`` for agent-C delivered to agent-B's chain would be
        # stored by B (cross-talk). The recipient stays authoritative over the
        # transport that happened to carry it.
        if env.recipient not in ("*", self.agent_id):
            return
        for handler in list(self._handlers):
            try:
                handler(env, sender_peer)
            except Exception:
                logger.warning("chain handler failed", exc_info=True)

    def _bump(self, key: str, amount: int = 1) -> None:
        """Thread-safe stats counter; transport threads call this concurrently."""
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + int(amount)

    # -- operator surface -----------------------------------------------------------

    def health(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        out = []
        for transport in self._transports:
            circuit = self._circuits[transport.name]
            out.append({
                "transport": transport.name,
                "profile": transport.profile.name,
                "breaker_open": circuit.open_until > now,
                "consecutive_failures": circuit.failures,
                "latency_ewma_s": circuit.latency_ewma,
                "accepts": [cls for cls in range(4)
                            if transport.accepts_class(cls)],
            })
        return out

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.append(event_type, payload)
        except Exception:
            pass


