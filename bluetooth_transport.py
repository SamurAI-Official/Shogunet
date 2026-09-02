"""
Bluetooth transport for Shogunet -- short-range links (Classic RFCOMM and BLE).

Two backends:
- RFCOMM (Classic Bluetooth): stream transport over ``AF_BLUETOOTH`` sockets
  (Linux/pybluez) or a Kotlin-bridge contract on Android. Framing is the
  same length-prefixed stream as TCP.
- BLE: datagram transport with the compact codec and segmentation. The
  default MTU is 23 bytes (negotiable to 247), so segmentation is the norm.

Stub mode (default): in-process pipe for tests and sim, no radios.

Discovery on Bluetooth is pairing-based -- the transport does not scan.
Pairing is the consent act; the agent_registry allowlist is enforced at
connection time.
"""

import logging
import struct
import threading
import time
from collections import deque
from typing import IO, Optional, Tuple

from protocol import (
    CLASS_BULK, CODEC_COMPACT, MAX_ENVELOPE_BYTES, Envelope, ProtocolError,
    SegmentAssembler, decode, encode, segment_frame,
)
from security import sanitize_text
from transports import BaseTransport, profile_for

logger = logging.getLogger(__name__)

_LEN_FMT = ">I"
_LEN_SIZE = 4


class _BytePipe:
    """In-process full-duplex byte pipe (shared stub with LoRa tests)."""

    __slots__ = ("_in", "_out", "_lock", "_closed")

    def __init__(self):
        self._in: deque[bytes] = deque()
        self._out: Optional["_BytePipe"] = None
        self._lock = threading.Lock()
        self._closed = False

    def pair(self, other: "_BytePipe") -> None:
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


class BluetoothTransport(BaseTransport):
    """Short-range transport for Bluetooth Classic (RFCOMM) and BLE.

    ``mode`` selects the framing strategy:
    - ``"rfcomm"``: length-prefixed stream (like TCP).
    - ``"ble"``: datagram with segmentation, compact codec, tiny MTU.
    """

    name = "bluetooth"

    def __init__(self, agent_id: str, port: Optional[IO[bytes]] = None,
                 mode: str = "ble", profile: str = "bluetooth",
                 max_queue: int = 64, chunk_size: int = 512,
                 ble_mtu: int = 23, audit: Optional[object] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        if mode not in ("rfcomm", "ble"):
            raise ValueError(f"unknown bluetooth mode '{mode}'")
        self._port = port
        self._mode = mode
        self._profile = profile_for(profile)
        self._max_queue = int(max_queue)
        self._outbox: deque[tuple[bytes, float]] = deque()
        self._chunk_size = int(chunk_size)
        self._ble_mtu = int(ble_mtu)
        self._audit = audit
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._assembler = SegmentAssembler(max_pending=16, ttl_s=30.0)
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
            raise RuntimeError("BluetoothTransport has no port")
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="bt-rx", daemon=True)
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
        if cls >= CLASS_BULK:
            logger.debug("bt: dropping class %d frame (short-range link)", cls)
            return False
        with self._lock:
            if len(self._outbox) >= self._max_queue:
                logger.warning("bt: outbox full, dropping frame")
                return False
            self._outbox.append((bytes(frame), time.monotonic()))
        return True

    def _loop(self) -> None:
        while self._running:
            try:
                self._pump_receiver()
                self._drain_outbox()
                time.sleep(0.01)
            except Exception as exc:
                logger.error("bt: loop error: %s", exc)
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
            logger.error("bt: read error: %s", exc)
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    pass
            return
        if not chunk:
            return
        self._recv_buf += chunk
        if self._mode == "rfcomm":
            self._pump_rfcomm()
        else:
            self._pump_ble()

    def _pump_rfcomm(self) -> None:
        while len(self._recv_buf) >= _LEN_SIZE:
            (length,) = struct.unpack_from(_LEN_FMT, self._recv_buf, 0)
            if length > MAX_ENVELOPE_BYTES:
                logger.warning("bt: oversized frame length %d, draining", length)
                self._recv_buf.clear()
                return
            if length + _LEN_SIZE > len(self._recv_buf):
                return
            frame = bytes(self._recv_buf[_LEN_SIZE:length + _LEN_SIZE])
            del self._recv_buf[:length + _LEN_SIZE]
            self._handle_frame(frame)

    def _pump_ble(self) -> None:
        while self._recv_buf:
            if len(self._recv_buf) < _LEN_SIZE:
                return
            (length,) = struct.unpack_from(_LEN_FMT, self._recv_buf, 0)
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
            logger.warning("bt: decode error: %s", exc)
            return
        self._peers.add(env.sender)
        if self._audit is not None:
            try:
                self._audit.append("bt_recv", {
                    "sender": env.sender, "type": env.msg_type,
                    "msg_id": env.msg_id, "bytes": len(frame),
                })
            except Exception:
                pass
        for cb in list(self._callbacks):
            try:
                cb(env.sender, assembled)
            except Exception:
                logger.warning("bt: subscriber error", exc_info=True)
        if self._on_recv is not None:
            try:
                self._on_recv(env, env.sender)
            except Exception as exc:
                logger.error("bt: recv callback error: %s", exc)

    def _drain_outbox(self) -> None:
        with self._lock:
            if not self._outbox:
                return
            frame, _ = self._outbox[0]
        if self._mode == "ble":
            chunks = segment_frame(frame, self._ble_mtu)
        else:
            chunks = [frame]
        for chunk in chunks:
            wire = struct.pack(_LEN_FMT, len(chunk)) + chunk
            try:
                self._port.write(wire)
            except Exception as exc:
                logger.error("bt: write error: %s", exc)
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
                self._audit.append("bt_send", {"bytes": len(frame), "chunks": len(chunks)})
            except Exception:
                pass

    def send_envelope(self, env: Envelope) -> bool:
        frame = encode(env, codec=CODEC_COMPACT)
        return self.send_frame(frame)

    def peers(self) -> set:
        return set(self._peers)

    def on_recv(self, callback) -> None:
        self._on_recv = callback

    def on_error(self, callback) -> None:
        self._on_error = callback

    def close(self) -> None:
        self._closed = True


def create_bluetooth_pipe_pair() -> Tuple[_BytePipe, _BytePipe]:
    """Return two ends of an in-process Bluetooth link (tests/sim)."""
    a, b = _BytePipe(), _BytePipe()
    a.pair(b)
    b.pair(a)
    return a, b
