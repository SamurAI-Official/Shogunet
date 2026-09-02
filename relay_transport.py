"""
Shogunet relay transport (5G / 4G / EDGE, cross-NAT)
=====================================================

Client half of the relay hub path for fleets unreachable by inbound
connections (cellular CGNAT, office NAT). On ``start`` the transport
registers with the hub, then a single background poll thread long-polls the
per-agent mailbox and drains inbound frames into a bounded inbox that
``poll()`` dispatches to subscribers -- the same no-thread-enters-user-code
discipline as every other transport.

The hub keeps its own mailbox bounded and TTL-expired; dedup is handled at
the agent layer by the fallback chain's ``InboxDedup``, so at-least-once
delivery over a lossy poll loop stays idempotent.
"""

import base64
import json
import logging
import threading
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import quote
from protocol import MAX_ENVELOPE_BYTES, ProtocolError, peek_header
from security import sanitize_text
from transports import BaseTransport, profile_for

logger = logging.getLogger(__name__)

DEFAULT_POLL_TIMEOUT_MS = 30000.0
_HUB_REQUEST_TIMEOUT_S = 60.0


class _RelayHttpClient:
    """Tiny stdlib HTTP/JSON client (no ``requests`` dependency)."""

    def __init__(self, base_url: str, timeout_s: float = _HUB_REQUEST_TIMEOUT_S):
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = max(1.0, float(timeout_s))

    def request(self, method: str, path: str,
                body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = resp.read().decode("utf-8")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise ConnectionError(f"relay request failed: {exc}") from exc
        return json.loads(payload or "{}")

    def register(self, agent_id: str, realm: str,
                 manifest: Optional[Dict[str, Any]]) -> bool:
        result = self.request("POST", "/register",
                              {"agent": agent_id, "realm": realm,
                               "manifest": manifest or {}})
        return bool(result.get("ok"))

    def deliver(self, recipient: str, frame: bytes) -> bool:
        if len(frame) > MAX_ENVELOPE_BYTES:
            return False
        result = self.request(
            "POST", "/deliver",
            {"recipient": recipient,
             "frame_b64": base64.b64encode(frame).decode("ascii")})
        return bool(result.get("ok"))

    def poll(self, agent_id: str, timeout_ms: float) -> List[Dict[str, Any]]:
        safe = quote(agent_id, safe="")
        path = f"/poll/{safe}?timeout_ms={int(timeout_ms)}"
        result = self.request("GET", path)
        if not result.get("ok"):
            return []
        return list(result.get("messages") or [])


class RelayTransport(BaseTransport):
    """Poll-based relay transport for cellular / NATed fleets."""

    name = "relay"
    profile = profile_for("4g")
    broadcast_supported = False  # the hub routes per-recipient mailboxes only

    def __init__(self, agent_id: str, hub_url: str,
                 realm: str = "*", manifest: Optional[Dict[str, Any]] = None,
                 profile: str = "4g",
                 poll_timeout_ms: float = DEFAULT_POLL_TIMEOUT_MS,
                 max_queue: int = 256, poll_retry_s: float = 1.0,
                 client: Optional[_RelayHttpClient] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        if not str(hub_url):
            raise ValueError("hub_url required")
        self.hub_url = str(hub_url)
        self.realm = sanitize_text(realm, 8) or "*"
        self.manifest = dict(manifest or {})
        self.profile = profile_for(profile)
        self.poll_timeout_ms = max(100.0, float(poll_timeout_ms))
        self.poll_retry_s = max(0.1, float(poll_retry_s))
        self._http = client or _RelayHttpClient(self.hub_url)
        self._inbox: Deque[Tuple[str, bytes]] = deque(
            maxlen=max(1, int(max_queue)))
        self._callbacks: List[Callable[[str, bytes], None]] = []
        self._lock = threading.RLock()
        self._running = False
        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._connected = False
        self._stats = {"registered": 0, "sent": 0, "send_failed": 0,
                       "received": 0, "poll_errors": 0}

# -- lifecycle -----------------------------------------------------------

    def is_available(self) -> bool:
        return self._running and self._connected

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop.clear()
        self._connected = self._http.register(self.agent_id, self.realm,
                                              self.manifest)
        if self._connected:
            self._stats["registered"] += 1
            logger.info("relay registered %s at %s",
                        self.agent_id, self.hub_url)
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="sgn-relay-poll", daemon=True)
        self._poll_thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        self._stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        self._connected = False

    def _poll_loop(self) -> None:
        retry = self.poll_retry_s
        while not self._stop.is_set():
            if not self._connected:
                try:
                    ok = self._http.register(self.agent_id, self.realm,
                                             self.manifest)
                    if ok:
                        self._stats["registered"] += 1
                        self._connected = True
                except ConnectionError:
                    pass
                if not self._connected:
                    if self._stop.wait(retry):
                        return
                    retry = min(retry * 1.5, 30.0)
                    continue
            try:
                messages = self._http.poll(self.agent_id,
                                           self.poll_timeout_ms)
            except ConnectionError:
                with self._lock:
                    self._stats["poll_errors"] += 1
                self._connected = False
                if self._stop.wait(retry):
                    return
                retry = min(retry * 1.5, 30.0)
                continue
            retry = self.poll_retry_s
            for message in messages:
                try:
                    frame = base64.b64decode(
                        str(message.get("frame_b64", "")), validate=True)
                except (TypeError, ValueError):
                    with self._lock:
                        self._stats["poll_errors"] += 1
                    continue
                try:
                    peek_header(frame)
                except ProtocolError:
                    with self._lock:
                        self._stats["poll_errors"] += 1
                    continue
                sender = sanitize_text(message.get("sender", ""), 48) or "?"
                with self._lock:
                    if len(self._inbox) == self._inbox.maxlen:
                        self._inbox.popleft()     # bounded: newest wins
                    self._inbox.append((sender, frame))

    # -- send / receive -------------------------------------------------------

    def send_frame(self, peer: Optional[str], frame: bytes) -> bool:
        frame = bytes(frame)
        if not self._connected or peer in (None, "", "*"):
            self._stats["send_failed"] += 1
            return False
        try:
            peek_header(frame)
        except ProtocolError:
            self._stats["send_failed"] += 1
            return False
        try:
            ok = self._http.deliver(str(peer), frame)
        except ConnectionError:
            ok = False
            self._connected = False
        self._stats["sent" if ok else "send_failed"] += 1
        return ok

    def subscribe(self, callback: Callable[[str, bytes], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

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
                except Exception:
                    logger.warning("relay callback failed", exc_info=True)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)

# [SGN:CONT]