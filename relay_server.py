"""
Shogunet relay hub (5G / 4G / EDGE cross-NAT path)
===================================================

A minimal in-memory rendezvous for agents that cannot accept inbound
connections (cellular CGNAT, firewalled fleets). Agents register, poll
their mailbox with long-polling, and publish by posting to a recipient's
mailbox.

Design
------
- Per-agent mailboxes are in-memory and bounded (``max_messages`` per
  agent, ``message_ttl_s`` expiry). The hub is a dumb pipe: it never
  persists plaintext and carries no auth beyond protocol-level validation.
- Trust lives at the agent layer (``AgentRegistry`` pairing = consent);
  the hub only relocates bytes. Every frame is validated with
  ``peek_header`` / ``decode`` and size-capped before enqueue, and the
  envelope's ``sender`` must match a registered agent -- spoofed frames
  are refused.
- Long-poll ``GET /poll/{agent_id}?timeout_ms=N`` blocks until a message
  or the timeout, so cellular agents get low-latency delivery without
  holding a permanent socket (CGNAT keeps working because delivery is
  pull-based).

HTTP API
--------
POST /register  {"agent": id, "realm": "*", "manifest": {...}}
POST /deliver   {"recipient": id, "frame_b64": "..."}
GET  /poll/{id}?timeout_ms=N  -> {"messages":[{"sender","frame_b64","msg_id"}]}
GET  /health    -> {"ok": true}
"""

import base64
import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from protocol import MAX_ENVELOPE_BYTES, ProtocolError, decode, peek_header
from security import sanitize_text

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES = 512
DEFAULT_MESSAGE_TTL_S = 300.0
POLL_MAX_MESSAGES = 64
MAX_AGENT_ID = 48


class _Mailbox:
    """One agent's inbound queue plus a wake condition for long-polling."""

    __slots__ = ("messages", "cv")

    def __init__(self) -> None:
        self.messages: Deque[Tuple[float, str, bytes]] = deque()
        self.cv = threading.Condition()


class RelayHub:
    """In-memory relay controller (socket-free; unit-testable directly)."""

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES,
                 message_ttl_s: float = DEFAULT_MESSAGE_TTL_S,
                 max_registered: int = 1024):
        self.max_messages = max(1, int(max_messages))
        self.message_ttl_s = max(0.01, float(message_ttl_s))   # testable small TTLs
        self.max_registered = max(1, int(max_registered))
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._mailboxes: Dict[str, _Mailbox] = {}
        self._lock = threading.RLock()
        self._stats = {"registered": 0, "delivered": 0, "polls": 0,
                       "dropped_bad": 0, "dropped_spoofed": 0,
                       "dropped_unknown_recipient": 0, "dropped_full": 0,
                       "expired": 0}

    # -- registration --------------------------------------------------------------

    def register(self, agent_id: str, realm: str = "*",
                 manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Admit an agent to the hub; returns its relay handle."""
        agent = sanitize_text(agent_id, MAX_AGENT_ID).strip()
        if not agent:
            return {"ok": False, "reason": "agent_id required"}
        with self._lock:
            if agent not in self._agents and len(self._agents) >= self.max_registered:
                return {"ok": False, "reason": "hub at capacity"}
            self._agents[agent] = {
                "realm": sanitize_text(realm, 8) or "*",
                "manifest": dict(manifest or {}),
                "registered_at": time.time(),
            }
            if agent not in self._mailboxes:
                self._mailboxes[agent] = _Mailbox()
            self._stats["registered"] += 1
        return {"ok": True, "agent": agent}

    def is_registered(self, agent_id: str) -> bool:
        with self._lock:
            return str(agent_id) in self._agents

    def agents(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {agent: dict(info) for agent, info in self._agents.items()}

    # -- delivery ----------------------------------------------------------------

    def deliver(self, recipient: str, frame: bytes) -> Dict[str, Any]:
        """Validate and enqueue one frame for a recipient's mailbox."""
        frame = bytes(frame)
        try:
            head = peek_header(frame)               # cheap validation
            env = decode(frame)                     # sender enforcement
        except ProtocolError:
            self._stats["dropped_bad"] += 1
            return {"ok": False, "reason": "invalid frame"}
        sender = env.sender
        with self._lock:
            if sender not in self._agents:
                self._stats["dropped_spoofed"] += 1
                return {"ok": False, "reason": "unknown sender"}
            mailbox = self._mailboxes.get(str(recipient))
            if mailbox is None:
                self._stats["dropped_unknown_recipient"] += 1
                return {"ok": False, "reason": "recipient unknown"}
            self._expire_locked(mailbox)
            if len(mailbox.messages) >= self.max_messages:
                self._stats["dropped_full"] += 1
                return {"ok": False, "reason": "recipient mailbox full"}
            mailbox.messages.append((time.monotonic() + self.message_ttl_s,
                                     sender, frame))
        with mailbox.cv:
            mailbox.cv.notify_all()
        with self._lock:
            self._stats["delivered"] += 1
        return {"ok": True, "sender": sender}

    def poll(self, agent_id: str, timeout_ms: float = 30000.0,
             max_messages: int = POLL_MAX_MESSAGES) -> Dict[str, Any]:
        """Long-pull a registered agent's mailbox (blocks up to timeout_ms).

        Returns a JSON-safe dict of drained messages. Stale entries are
        expired greedily so a mailbox can never grow without bound.
        """
        agent = sanitize_text(agent_id, MAX_AGENT_ID).strip()
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)
        with self._lock:
            mailbox = self._mailboxes.get(agent)
        if mailbox is None:
            return {"ok": False, "messages": [], "reason": "unregistered"}
        deadline = time.monotonic() + timeout_s
        drained: List[Tuple[float, str, bytes]] = []
        with mailbox.cv:
            while True:
                now = time.monotonic()
                with self._lock:
                    self._expire_locked(mailbox)
                    while mailbox.messages and len(drained) < max_messages:
                        drained.append(mailbox.messages.popleft())
                if drained or now >= deadline:
                    break
                mailbox.cv.wait(timeout=max(0.01, deadline - now))
        with self._lock:
            self._stats["polls"] += 1
        messages = [{"sender": sender,
                     "frame_b64": base64.b64encode(frame).decode("ascii"),
                     "msg_id": peek_header(frame)["msg_id"]}
                    for _expiry, sender, frame in drained]
        return {"ok": True, "messages": messages}

    # -- maintenance -----------------------------------------------------------

    def _expire_locked(self, mailbox: _Mailbox) -> int:
        """Drop messages older than the TTL; returns the count expired."""
        if not mailbox.messages:
            return 0
        now = time.monotonic()
        expired = 0
        while mailbox.messages and mailbox.messages[0][0] < now:
            mailbox.messages.popleft()
            expired += 1
        if expired:
            self._stats["expired"] += expired
        return expired

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

# ---------------------------------------------------------------------------
# HTTP server layer
# ---------------------------------------------------------------------------

def make_handler(hub: RelayHub):
    """Build a BaseHTTPRequestHandler subclass bound to a RelayHub.

    ``hub`` is attached as a class attribute after class creation -- a class
    body cannot close over enclosing function locals on an assignment RHS.
    """

    class _Handler(BaseHTTPRequestHandler):
        hub = None   # set below

        def log_message(self, fmt, *args):     # silence the access log
            return

        def _json(self, code: int, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Optional[Dict[str, Any]]:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                return None
            if length <= 0 or length > 64 * 1024:
                return None
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None

        def _split_path(self):
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            return parts, query

        # -- POST -------------------------------------------------------------

        def do_POST(self) -> None:
            parts, _ = self._split_path()
            body = self._read_json()
            if not isinstance(body, dict):
                self._json(400, {"ok": False, "reason": "bad json"})
                return
            if parts and parts[0] == "register":
                result = self.hub.register(
                    str(body.get("agent", "")),
                    realm=str(body.get("realm", "*")),
                    manifest=body.get("manifest"))
                self._json(200 if result["ok"] else 409, result)
                return
            if parts and parts[0] == "deliver":
                recipient = str(body.get("recipient", ""))
                try:
                    frame = base64.b64decode(str(body.get("frame_b64", "")),
                                             validate=True)
                except (TypeError, ValueError):
                    self._json(400, {"ok": False, "reason": "bad frame_b64"})
                    return
                result = self.hub.deliver(recipient, frame)
                self._json(200 if result["ok"] else 412, result)
                return
            self._json(404, {"ok": False, "reason": "not found"})

        # -- GET ----------------------------------------------------------------

        def do_GET(self) -> None:
            parts, query = self._split_path()
            if parts and parts[0] == "poll" and len(parts) == 2:
                agent = parts[1]
                try:
                    timeout_ms = float(query.get("timeout_ms", "30000"))
                except ValueError:
                    timeout_ms = 30000.0
                result = self.hub.poll(agent, timeout_ms=timeout_ms)
                self._json(200 if result["ok"] else 404, result)
                return
            if parts and parts[0] == "health":
                self._json(200, {"ok": True, "agents": len(self.hub.agents())})
                return
            self._json(404, {"ok": False, "reason": "not found"})

    _Handler.hub = hub
    return _Handler


def run_relay(hub: RelayHub, host: str = "127.0.0.1", port: int = 0,
              daemon: bool = True):
    """Bind a ThreadingHTTPServer around a hub; returns (server, thread).

    Each long-poll request occupies one worker thread, so ThreadingHTTPServer
    is the right stdlib choice for a modest fleet; a production hub would use
    asyncio/aiohttp for large fan-out.
    """
    server = ThreadingHTTPServer((host, port), make_handler(hub))
    thread = threading.Thread(target=server.serve_forever,
                              name="sgn-relay-http", daemon=daemon)
    thread.start()
    return server, thread