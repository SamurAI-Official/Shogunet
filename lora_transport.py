"""
LoRa transport for Shogunet -- constrained long-range links (0.3-27 kbit/s,
duty-cycled, ~10 km practical).

Two backends, both optional extras:
- ``pyserial`` + an SX126x/SX127x module in P2P (AT/command) mode: real radio.
- Stub mode (default): in-process byte pipe for tests and sim, no hardware.

Wire framing mirrors the TCP transport: ``[4-byte big-endian length][SGN frame]``.
The transport always uses the compact codec (every byte is airtime) and
enforces the LoRa link profile's tiny payload cap. Frames exceeding the
cap are segmented via ``protocol.segment_frame`` and reassembled via
``SegmentAssembler``.

Priority gating: only P0 (control) and P1 (memory delta) are eligible on
constrained links -- P2/P3 are refused at send time with a logged drop, so
the fallback chain can route them to a better link. This is the mechanism
that keeps the codependent memory mesh working over LoRa: ``reinforce`` and
``memory_digest`` messages are a few dozen bytes each.
"""

import logging
import struct
import threading
import time
from collections import deque
from typing import IO, Optional, Tuple

from protocol import (
    CLASS_TASK, CODEC_COMPACT, MAX_ENVELOPE_BYTES, Envelope, ProtocolError,
    SegmentAssembler, decode, encode, segment_frame,
)
from security import sanitize_text
from transports import BaseTransport, profile_for

logger = logging.getLogger(__name__)

_LEN_FMT = ">I"
_LEN_SIZE = 4


class _SerialPipe:
    """In-process full-duplex byte pipe standing in for a serial port.

    Used by tests and sim so LoRa framing, segmentation and priority gating
    are exercised without hardware. Supports ``read``/``write``/``close``.
    """

    __slots__ = ("_in", "_out", "_lock", "_closed")

    def __init__(self):
        self._in: deque[bytes] = deque()
        self._out: Optional["_SerialPipe"] = None
        self._lock = threading.Lock()
        self._closed = False

    def pair(self, other: "_SerialPipe") -> None:
        self._out = other

    def write(self, data: bytes) -> int:
        if self._closed:
            raise OSError("pipe closed")
        with self._lock:
            if self._out is None or self._out._closed:
                return 0
            self._out._in.append(bytes(data))
        return len(data)

    def read(self, n: int = 0) -> bytes:
        if self._closed:
            return b""
        with self._lock:
            if not self._in:
                return b""
            if n <= 0:
                joined = b"".join(self._in)
                self._in.clear()
                return joined
            buf = bytearray()
            while self._in and len(buf) < n:
                chunk = self._in[0]
                need = n - len(buf)
                if len(chunk) <= need:
                    buf += chunk
                    self._in.popleft()
                else:
                    buf += chunk[:need]
                    self._in[0] = chunk[need:]
            return bytes(buf)

    def close(self) -> None:
        self._closed = True


class LoraTransport(BaseTransport):
    """Constrained transport for LoRa links (SX-series P2P or stub pipe)."""

    name = "lora"

    def __init__(self, agent_id: str, port: Optional[IO[bytes]] = None,
                 profile: str = "lora", max_queue: int = 64,
                 chunk_size: int = 120, duty_cycle_window_s: float = 3600.0,
                 max_airtime_fraction: float = 0.01, audit: Optional[object] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self._port = port
        self._profile = profile_for(profile)
        self._chunk_size = int(chunk_size)
        if self._chunk_size < 40:
            raise ValueError("chunk_size too small for SGN framing")
        self._max_queue = int(max_queue)
        self._outbox: deque[tuple[bytes, float]] = deque()
        self._duty_window = float(duty_cycle_window_s)
        self._max_airtime = float(max_airtime_fraction)
        self._airtime_used = 0.0
        self._window_start = time.monotonic()
        self._audit = audit
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._assembler = SegmentAssembler(max_pending=8, ttl_s=30.0)
        self._recv_buf = bytearray()
        self._on_recv = None
        self._on_error = None
        self._callbacks: list = []
        self._peers: set = set()

    def is_available(self) -> bool:
        return self._port is not None

    def start(self) -> None:
        if self._running:
            return
        if self._port is None:
            raise RuntimeError("LoraTransport has no port")
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="lora-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass

    def poll(self, timeout: float = 0.1) -> None:
        if self._port is None:
            return
        self._pump_receiver()
        self._drain_outbox()

    def subscribe(self, callback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def send_frame(self, peer=None, frame=None) -> bool:
        # Accept both call styles: (frame) for direct use and (peer, frame)
        # for the fallback chain, which calls transport.send_frame(recipient, frame).
        if frame is None:
            frame, peer = peer, None
        if len(frame) < 16:
            return False
        cls = frame[5]
        if cls >= CLASS_TASK:
            logger.debug("lora: dropping class %d frame (constrained link)", cls)
            return False
        with self._lock:
            if len(self._outbox) >= self._max_queue:
                logger.warning("lora: outbox full, dropping frame")
                return False
            self._outbox.append((bytes(frame), time.monotonic()))
        return True

    def on_recv(self, callback) -> None:
        self._on_recv = callback

    def _reset_window_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= self._duty_window:
            self._window_start = now
            self._airtime_used = 0.0

    def _airtime_budget_remaining(self) -> float:
        self._reset_window_if_needed()
        return max(0.0, self._max_airtime * self._duty_window - self._airtime_used)

    def _consume_airtime(self, byte_count: int) -> None:
        kbps = self._profile.kbps if self._profile.kbps > 0 else 5.0
        seconds = (byte_count * 8) / (kbps * 1000)
        self._airtime_used += seconds

    def _loop(self) -> None:
        while self._running:
            try:
                self._pump_receiver()
                self._drain_outbox()
                time.sleep(0.01)
            except Exception as exc:
                logger.error("lora: loop error: %s", exc)
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass

    def _pump_receiver(self) -> None:
        if self._port is None:
            return
        try:
            chunk = self._port.read(256)
        except Exception as exc:
            logger.error("lora: read error: %s", exc)
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    pass
            return
        if not chunk:
            return
        self._recv_buf += chunk
        while len(self._recv_buf) >= _LEN_SIZE:
            (length,) = struct.unpack_from(_LEN_FMT, self._recv_buf, 0)
            if length > MAX_ENVELOPE_BYTES:
                logger.warning("lora: oversized frame length %d, draining", length)
                self._recv_buf.clear()
                return
            if length + _LEN_SIZE > len(self._recv_buf):
                return
            frame = bytes(self._recv_buf[_LEN_SIZE:length + _LEN_SIZE])
            del self._recv_buf[:length + _LEN_SIZE]
            self._handle_frame(frame)

    def _handle_frame(self, frame: bytes) -> None:
        assembled = self._assembler.add(frame)
        if assembled is None:
            return
        try:
            env = decode(assembled)
        except ProtocolError as exc:
            logger.warning("lora: decode error: %s", exc)
            return
        self._peers.add(env.sender)
        if self._audit is not None:
            try:
                self._audit.append("lora_recv", {
                    "sender": env.sender, "type": env.msg_type,
                    "msg_id": env.msg_id, "bytes": len(frame),
                })
            except Exception:
                pass
        for cb in list(self._callbacks):
            try:
                cb(env.sender, assembled)
            except Exception:
                logger.warning("lora: subscriber error", exc_info=True)
        if self._on_recv is not None:
            try:
                self._on_recv(env, env.sender)
            except Exception as exc:
                logger.error("lora: recv callback error: %s", exc)

    def _drain_outbox(self) -> None:
        with self._lock:
            if not self._outbox:
                return
            frame, _ = self._outbox[0]
        chunks = segment_frame(frame, self._chunk_size)
        for chunk in chunks:
            budget = self._airtime_budget_remaining()
            kbps = self._profile.kbps if self._profile.kbps > 0 else 5.0
            needed = (len(chunk) + _LEN_SIZE) * 8 / (kbps * 1000)
            if needed > budget:
                return
            self._consume_airtime(len(chunk) + _LEN_SIZE)
            wire = struct.pack(_LEN_FMT, len(chunk)) + chunk
            try:
                self._port.write(wire)
            except Exception as exc:
                logger.error("lora: write error: %s", exc)
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass
                return
        with self._lock:
            self._outbox.popleft()
        if self._audit is not None:
            try:
                self._audit.append("lora_send", {"bytes": len(frame), "chunks": len(chunks)})
            except Exception:
                pass

    def send_envelope(self, env: Envelope) -> bool:
        frame = encode(env, codec=CODEC_COMPACT)
        return self.send_frame(frame)

    def peers(self) -> set:
        return set(self._peers)

    def on_error(self, callback) -> None:
        self._on_error = callback

    def close(self) -> None:
        self._closed = True


def create_pipe_pair() -> Tuple[_SerialPipe, _SerialPipe]:
    """Return two ends of an in-process LoRa link (tests/sim)."""
    a, b = _SerialPipe(), _SerialPipe()
    a.pair(b)
    b.pair(a)
    return a, b
