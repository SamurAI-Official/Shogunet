"""
Length-prefixed TCP transport for IP-class links (WiFi, WiFi-Halow, and the
LAN side of cellular deployments).

Wire format per frame: ``[4-byte big-endian length][SGN frame]``.

Handshake: the first frame on a new connection MUST be an ``announce``
envelope identifying the sender. Until that handshake completes, every
further byte from the connection is refused and counted -- no anonymous
data ever reaches the agent. The announce also teaches the listener the
dialer's listen port, so both sides can dial each other after a single
outbound connection.

Connection-oriented transports ignore the LinkProfile's datagram chunk cap
for framing, but still refuse frames beyond ``MAX_ENVELOPE_BYTES``.
"""

import logging
import socket
import struct
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from protocol import (CODEC_COMPACT, MAX_ENVELOPE_BYTES, Envelope,
                      ProtocolError, decode, encode, new_msg_id, peek_header)
from security import sanitize_text
from transports import BaseTransport, profile_for

logger = logging.getLogger(__name__)

_LEN_FMT = ">I"
_LEN_SIZE = 4


class _Connection:
    """One established TCP peer connection (pre- or post-handshake)."""

    __slots__ = ("sock", "addr", "agent_id", "client_version", "lock")

    def __init__(self, sock: socket.socket, addr: Tuple[str, int]):
        self.sock = sock
        self.addr = addr
        self.agent_id: Optional[str] = None
        self.client_version: str = ""
        self.lock = threading.Lock()


class TCPTransport(BaseTransport):
    """Stream transport for broadband and midband IP networks."""

    name = "tcp"

    def __init__(self, agent_id: str, listen: bool = True,
                 listen_host: str = "127.0.0.1", listen_port: int = 0,
                 profile: str = "wifi", max_queue: int = 256,
                 connect_timeout_s: float = 5.0, audit: Optional[object] = None,
                 admission_check: Optional[Callable[[str], bool]] = None,
                 on_connection_lost: Optional[Callable[[str], None]] = None,
                 version_check: Optional[Callable[[str], bool]] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self.profile = profile_for(profile)
        self.audit = audit
        # Pairing = consent: when set, the handshake admits only agents the
        # hook accepts (the host passes AgentRegistry.is_paired).
        self._admission_check = admission_check
        # Optional callback fired with the agent_id when a handshaken
        # connection is torn down. Used by the host to detect peer loss
        # immediately instead of waiting for heartbeat TTL expiry.
        self._on_connection_lost = on_connection_lost
        # Version handshake: when set, the announce's ``shugonet_version``
        # must pass this hook (the host passes version.is_compatible) or the
        # connection is refused and audited. Peers that send no version
        # (pre-0.4.0) are admitted but reported as version "unknown".
        self._version_check = version_check
        self._listen = bool(listen)
        self._listen_host = str(listen_host)
        self._listen_port = max(0, min(65535, int(listen_port)))
        self._connect_timeout = max(0.5, float(connect_timeout_s))
        self._server: Optional[socket.socket] = None
        self._peers: Dict[str, Tuple[str, int]] = {}
        self._conns: Dict[str, _Connection] = {}
        self._awaiting: Dict[str, _Connection] = {}
        self._inbox: Deque[Tuple[str, bytes]] = deque(maxlen=max(1, int(max_queue)))
        self._callbacks: List[Callable[[str, bytes], None]] = []
        self._state_lock = threading.RLock()
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._announce_frame: Optional[bytes] = None
        self._default_peer: Optional[str] = None
        self._stats = {"sent": 0, "send_failed": 0, "received": 0, "dropped": 0,
                       "handshake_refused": 0, "connections": 0, "dials": 0}

    # -- lifecycle ---------------------------------------------------------------

    def _build_announce(self) -> None:
        import version
        env = Envelope(msg_id=new_msg_id(), msg_type="announce",
                       sender=self.agent_id, recipient="*",
                       payload={"port": self.local_port,
                                "shugonet_version": version.VERSION})
        self._announce_frame = encode(env, CODEC_COMPACT)

    def is_available(self) -> bool:
        return self._running

    def peer_versions(self) -> Dict[str, str]:
        """shugonet version reported by each handshaken peer's announce."""
        with self._state_lock:
            return {aid: conn.client_version
                    for aid, conn in self._conns.items() if aid}

    def start(self) -> None:
        with self._state_lock:
            if self._running:
                return
            if self._listen:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self._listen_host, self._listen_port))
                server.listen(8)
                server.settimeout(0.25)
                self._server = server
            # Build the announce AFTER binding so it advertises the real
            # (possibly ephemeral) listen port to the fleet.
            self._build_announce()
            self._running = True
        if self._listen:
            self._accept_thread = threading.Thread(
                target=self._accept_loop, name="sgn-tcp-accept", daemon=True)
            self._accept_thread.start()

    def stop(self) -> None:
        with self._state_lock:
            self._running = False
        # Join outside the lock: the accept thread needs the state lock to
        # observe shutdown, and joining while holding it would deadlock.
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        with self._state_lock:
            server, self._server = self._server, None
            conns = list(self._conns.values()) + list(self._awaiting.values())
            self._conns.clear()
            self._awaiting.clear()
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        for conn in conns:
            self._close_conn(conn)

    @property
    def local_port(self) -> int:
        with self._state_lock:
            if self._server is not None:
                return int(self._server.getsockname()[1])
        return self._listen_port

    # -- peer management ---------------------------------------------------------

    def add_peer(self, agent_id: str, host: str, port: int) -> None:
        """Register a dialable peer address (from discovery or operator)."""
        agent = sanitize_text(agent_id, 48).strip()
        if not agent:
            raise ValueError("agent_id required")
        host = str(host)
        port = max(1, min(65535, int(port)))
        with self._state_lock:
            self._peers[agent] = (host, port)

    # -- connection internals ------------------------------------------------------

    def _close_conn(self, conn: _Connection) -> None:
        try:
            conn.sock.close()
        except OSError:
            pass
        with self._state_lock:
            if conn.agent_id and self._conns.get(conn.agent_id) is conn:
                self._conns.pop(conn.agent_id, None)
                # Notify listener (host) that this paired agent disconnected.
                if self._on_connection_lost is not None:
                    try:
                        self._on_connection_lost(conn.agent_id)
                    except Exception:
                        pass
            self._awaiting.pop(str(conn.addr), None)

    def _accept_loop(self) -> None:
        while True:
            with self._state_lock:
                running, server = self._running, self._server
            if not running or server is None:
                return
            try:
                sock, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn = _Connection(sock, addr)
            with self._state_lock:
                self._awaiting[str(addr)] = conn
            threading.Thread(target=self._read_loop, args=(conn,),
                             name="sgn-tcp-read", daemon=True).start()

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Read exactly n bytes; None on idle timeout, raise on hard error."""
        data = b""
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
            except socket.timeout:
                if not data and not self._running:
                    return None
                if not data:
                    return None
                continue
            if not chunk:
                raise ConnectionError("peer closed")
            data += chunk
        return data

    def _read_loop(self, conn: _Connection) -> None:
        sock = conn.sock
        try:
            sock.settimeout(0.25)
            while True:
                with self._state_lock:
                    running = self._running
                if not running:
                    return
                raw_len = self._recv_exact(sock, _LEN_SIZE)
                if raw_len is None:
                    continue
                (length,) = struct.unpack(_LEN_FMT, raw_len)
                if length <= 0 or length > MAX_ENVELOPE_BYTES:
                    raise ConnectionError("frame length out of range")
                frame = self._recv_exact(sock, length)
                if frame is None:
                    continue
                self._handle_frame(conn, frame)
        except (OSError, ConnectionError, ValueError) as exc:
            logger.debug("tcp connection %s ended: %s", conn.addr, exc)
        finally:
            self._close_conn(conn)

    def _handle_frame(self, conn: _Connection, frame: bytes) -> None:
        try:
            env = decode(frame)
        except ProtocolError:
            self._stats["dropped"] += 1
            return
        if conn.agent_id is None:
            # Handshake: the first frame MUST identify the sender.
            if env.msg_type != "announce" or not env.sender:
                self._stats["handshake_refused"] += 1
                self._audit("tcp_handshake_refused", {"addr": str(conn.addr)})
                raise ConnectionError("handshake refused")
            if self._admission_check is not None \
                    and not self._admission_check(env.sender):
                # Pairing = consent: unpaired agents are refused at the door
                # and their connection is torn down before any traffic flows.
                self._stats["handshake_refused"] += 1
                self._audit("tcp_admission_refused",
                            {"agent_id": env.sender, "addr": str(conn.addr)})
                raise ConnectionError("agent not paired")
            client_version = str(env.payload.get("shugonet_version", "") or "")
            if client_version and self._version_check is not None \
                    and not self._version_check(client_version):
                # Version handshake: an incompatible client is refused at the
                # door too -- protocol drift must never enter the fleet.
                self._stats["handshake_refused"] += 1
                import version
                self._audit("version_mismatch",
                            {"agent_id": env.sender, "addr": str(conn.addr),
                             "client_version": client_version[:16],
                             "host_version": version.VERSION})
                raise ConnectionError("incompatible shugonet version")
            conn.agent_id = env.sender
            conn.client_version = client_version or "unknown"
            port = env.payload.get("port")
            with self._state_lock:
                self._awaiting.pop(str(conn.addr), None)
                self._conns[conn.agent_id] = conn
                self._stats["connections"] += 1
                if isinstance(port, int) and 1 <= port <= 65535:
                    self._peers.setdefault(
                        conn.agent_id, (conn.addr[0], port))
        self._stats["received"] += 1
        with self._state_lock:
            if len(self._inbox) == self._inbox.maxlen:
                self._stats["dropped"] += 1
                self._inbox.popleft()
            self._inbox.append((conn.agent_id, frame))

    def _audit(self, event_type: str, payload: Dict[str, object]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.append(event_type, payload)
        except Exception:
            pass

    # -- send / receive ------------------------------------------------------------

    def _dial(self, agent_id: str) -> Optional[_Connection]:
        """Establish (or reuse) a handshaken connection to a known peer."""
        with self._state_lock:
            existing = self._conns.get(agent_id)
            if existing is not None:
                return existing
            peer = self._peers.get(agent_id)
            announce = self._announce_frame
        if peer is None or announce is None:
            return None
        try:
            sock = socket.create_connection(peer,
                                            timeout=self._connect_timeout)
        except OSError as exc:
            logger.debug("dial %s failed: %s", peer, exc)
            return None
        conn = _Connection(sock, peer)
        try:
            sock.settimeout(self._connect_timeout)
            with conn.lock:
                sock.sendall(struct.pack(_LEN_FMT, len(announce)) + announce)
            sock.settimeout(0.25)
        except OSError as exc:
            logger.debug("handshake send to %s failed: %s", peer, exc)
            try:
                sock.close()
            except OSError:
                pass
            return None
        conn.agent_id = agent_id
        with self._state_lock:
            self._conns[agent_id] = conn
            self._stats["dials"] += 1
            self._stats["connections"] += 1
        threading.Thread(target=self._read_loop, args=(conn,),
                         name="sgn-tcp-read", daemon=True).start()
        return conn

    def send_frame(self, peer: Optional[str] = None, frame: bytes = b"") -> bool:
        """Send a frame. Supports both call styles:

        - ``send_frame(frame)`` — used by the chain's ``send()`` path; sends to
          all connected peers (broadcast) or the default peer.
        - ``send_frame(peer, frame)`` — used by the chain's ``send_raw()`` path
          and by direct callers; sends to a specific peer.
        """
        # Handle both call styles: send_frame(frame) or send_frame(peer, frame)
        if frame == b"" and peer is not None and isinstance(peer, bytes):
            peer, frame = None, peer
        frame = bytes(frame)
        if len(frame) > MAX_ENVELOPE_BYTES:
            with self._state_lock:
                self._stats["send_failed"] += 1
            return False
        try:
            peek_header(frame)
        except ProtocolError:
            with self._state_lock:
                self._stats["send_failed"] += 1
            return False
        with self._state_lock:
            if peer in (None, "", "*"):
                targets = list(self._conns.values())
            else:
                conn = self._conns.get(str(peer))
                targets = [conn] if conn is not None else []
        if not targets and peer not in (None, "", "*"):
            conn = self._dial(str(peer))
            targets = [conn] if conn is not None else []
        # Fallback: if the requested peer is unknown (e.g. agent A sending to
        # agent B via the host, where the agent only knows the host as a
        # transport-level peer), send to the default peer (the host). The
        # host reads the envelope's recipient and forwards.
        if not targets and self._default_peer:
            conn = self._conns.get(self._default_peer)
            if conn is None:
                conn = self._dial(self._default_peer)
            targets = [conn] if conn is not None else []
        if not targets:
            with self._state_lock:
                self._stats["send_failed"] += 1
            return False
        payload = struct.pack(_LEN_FMT, len(frame)) + frame
        delivered = 0
        for conn in targets:
            try:
                with conn.lock:
                    conn.sock.sendall(payload)
                delivered += 1
            except OSError as exc:
                logger.debug("send to %s failed: %s", conn.addr, exc)
                self._close_conn(conn)
        with self._state_lock:
            if delivered == 0:
                self._stats["send_failed"] += 1
                ok = False
            else:
                self._stats["sent"] += 1
                ok = True
        return ok

    def subscribe(self, callback: Callable[[str, bytes], None]) -> None:
        with self._state_lock:
            self._callbacks.append(callback)

    def poll(self, timeout: float = 0.1) -> None:
        while True:
            with self._state_lock:
                if not self._inbox:
                    break
                sender, frame = self._inbox.popleft()
            for callback in list(self._callbacks):
                try:
                    callback(sender, frame)
                except Exception:   # subscriber bugs never kill the poll loop
                    logger.warning("tcp callback failed", exc_info=True)

    def stats(self) -> Dict[str, object]:
        with self._state_lock:
            return dict(self._stats)

    def has_active_connection(self) -> bool:
        """True if any handshaken connection exists (inbound or outbound).

        A connection is "active" only after the handshake completed and the
        peer's agent_id was assigned. Used by runtimes to detect admission
        refusal: if the server rejected the handshake, the read loop exits
        and clears the connection.
        """
        with self._state_lock:
            return bool(self._conns)



# [SGN:CONT]
