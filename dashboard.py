"""
Shogunet fleet dashboard (operator plane)
=========================================

A stdlib-only HTTP server attached to a ``ShugonetHost``:

    ShugoCore agents --> Shugonet host --> dashboard.py --+--> JSON REST API
                                                         +--> SSE event stream
                                                         +--> compiled SPA (dist/)

Endpoints:

- ``GET  /api/status``   -- host.status() snapshot (mode, roster, chain, mesh)
- ``GET  /api/audit``    -- audit chain tail + integrity verification
- ``GET  /api/events``   -- Server-Sent Events: live audit records + status ticks
- ``POST /api/pair``     -- {"agent_id", "manifest"?, "paired_by"?}
- ``POST /api/unpair``   -- {"agent_id"}
- ``POST /api/resume``   -- {"resumed_by"} (attribution required, audited)
- ``POST /api/broadcast``-- {"payload"} system broadcast to the fleet
- ``GET  /*``            -- the compiled SPA (shipped in shugonet_web/static),
                            falling back to index.html for client-side routes

Security: binds loopback by default; when ``token`` is set, every state-
changing POST (and the event stream) requires an ``X-Shugonet-Token`` or
``Authorization: Bearer`` header. All operator actions flow through the
host's audited surfaces -- the dashboard never touches internals directly.
"""

import json
import logging
import os
import queue
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

DEFAULT_DASHBOARD_PORT = 9002
_MAX_EVENTS = 500          # ring buffer of audit events for SSE replay
_STATUS_TICK_S = 2.0       # periodic status frame on the event stream
_MAX_BODY_BYTES = 65536

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".woff2": "font/woff2",
}


def _default_static_dir() -> Optional[str]:
    """Locate the compiled SPA: package data first, then a dev checkout."""
    try:
        import shugonet_web
        static = os.path.join(os.path.dirname(shugonet_web.__file__), "static")
        if os.path.isdir(static):
            return static
    except Exception:
        pass
    dev = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "shugonet_web", "static")
    return dev if os.path.isdir(dev) else None


class DashboardServer:
    """HTTP operator surface for one ShugonetHost."""

    def __init__(self, host: Any, port: int = DEFAULT_DASHBOARD_PORT,
                 bind: str = "127.0.0.1", token: Optional[str] = None,
                 static_dir: Optional[str] = None,
                 event_buffer: int = _MAX_EVENTS):
        self.host = host
        self.port = max(0, int(port))
        self.bind = str(bind or "127.0.0.1")
        self.token = str(token) if token else None
        self.static_dir = static_dir or _default_static_dir()
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max(16, int(event_buffer)))
        self._client_queues: List[queue.Queue] = []
        self._clients_lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._serve_thread: Optional[threading.Thread] = None
        self._broadcaster: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._audit = getattr(host, "audit", None)
        self._seed_events()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._server is not None:
            return
        server = ThreadingHTTPServer((self.bind, self.port), _DashboardHandler)
        server.daemon_threads = True
        server.dashboard = self                     # handler back-reference
        self._server = server
        if self._audit is not None and hasattr(self._audit, "subscribe"):
            self._audit.subscribe(self._on_audit_record)
        self._stop.clear()
        self._serve_thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.2},
            name="sgn-dashboard-serve", daemon=True)
        self._serve_thread.start()
        self._broadcaster = threading.Thread(
            target=self._broadcast_loop, name="sgn-dashboard-broadcast",
            daemon=True)
        self._broadcaster.start()
        port = server.server_address[1]
        logger.info("fleet dashboard on http://%s:%d", self.bind, port)

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        serve_thread = getattr(self, "_serve_thread", None)
        if serve_thread is not None and serve_thread.is_alive():
            serve_thread.join(timeout=2.0)
        self._serve_thread = None
        if self._audit is not None and hasattr(self._audit, "_subscribers"):
            try:
                self._audit._subscribers.remove(self._on_audit_record)
            except ValueError:
                pass
        with self._clients_lock:
            self._client_queues = []

    @property
    def url(self) -> Optional[str]:
        if self._server is None:
            return None
        return f"http://{self.bind}:{self._server.server_address[1]}"

    # -- event plumbing ---------------------------------------------------------

    def _seed_events(self) -> None:
        """Replay the audit file tail so a fresh dashboard has history."""
        if self._audit is None:
            return
        try:
            with open(self._audit.path, "r", encoding="utf-8") as handle:
                lines = deque(handle, maxlen=self._events.maxlen)
        except OSError:
            return
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("event"):
                self._events.append(record)

    def _on_audit_record(self, record: Dict[str, Any]) -> None:
        try:
            self._events.append(dict(record))
        except Exception:
            return
        self._fanout("audit", record)

    def _fanout(self, event: str, data: Dict[str, Any]) -> None:
        with self._clients_lock:
            clients = list(self._client_queues)
        for client in clients:
            try:
                client.put_nowait((event, data))
            except Exception:
                pass

    def _broadcast_loop(self) -> None:
        """Periodic status ticks keep the operator console honest even when
        the fleet is quiet (and double as SSE keepalives)."""
        while not self._stop.wait(_STATUS_TICK_S):
            try:
                self._fanout("status", self._status_snapshot())
            except Exception:
                logger.warning("dashboard status tick failed", exc_info=True)

    def register_client(self) -> queue.Queue:
        client: queue.Queue = queue.Queue(maxsize=256)
        with self._clients_lock:
            self._client_queues.append(client)
        return client

    def unregister_client(self, client: queue.Queue) -> None:
        with self._clients_lock:
            try:
                self._client_queues.remove(client)
            except ValueError:
                pass

    # -- API surface (called by the handler) --------------------------------------

    def _status_snapshot(self) -> Dict[str, Any]:
        try:
            return dict(self.host.status())
        except Exception:
            # A host mid-startup/teardown must not 500 the console.
            return {"mode": "unknown", "roster": [], "paired_count": 0,
                    "alive_count": 0, "error": "status unavailable"}

    def _audit_tail(self, limit: int = 200) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        if self._audit is not None:
            try:
                with open(self._audit.path, "r", encoding="utf-8") as handle:
                    lines = deque(handle, maxlen=max(1, min(2000, int(limit))))
            except OSError:
                lines = deque()
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return {"records": records}

    def _audit_problems(self) -> List[Dict[str, Any]]:
        if self._audit is None:
            return []
        try:
            return list(self._audit.verify())
        except Exception:
            return [{"reason": "verify failed"}]

    # -- operator actions ------------------------------------------------------------

    def op_pair(self, body: Dict[str, Any]) -> tuple:
        agent_id = str(body.get("agent_id") or "").strip()
        if not agent_id:
            return 400, {"ok": False, "reason": "agent_id required"}
        manifest = body.get("manifest") \
            if isinstance(body.get("manifest"), dict) else None
        paired_by = str(body.get("paired_by") or "dashboard")[:120]
        try:
            entry = self.host.pair(agent_id, manifest=manifest,
                                   paired_by=paired_by)
        except Exception as exc:
            return 400, {"ok": False, "reason": str(exc)}
        return 200, {"ok": True, "agent": entry}

    def op_unpair(self, body: Dict[str, Any]) -> tuple:
        agent_id = str(body.get("agent_id") or "").strip()
        if not agent_id:
            return 400, {"ok": False, "reason": "agent_id required"}
        removed = self.host.unpair(agent_id)
        return 200, {"ok": bool(removed), "removed": bool(removed)}

    def op_resume(self, body: Dict[str, Any]) -> tuple:
        # Attribution is the whole point of the resume path: an operator
        # bringing a fleet out of a latch must be on the audit record.
        resumed_by = str(body.get("resumed_by") or "").strip()
        if not resumed_by:
            return 400, {"ok": False,
                         "reason": "resumed_by attribution required"}
        self.host.resume(resumed_by)
        return 200, {"ok": True, "resumed_by": resumed_by}

    def op_broadcast(self, body: Dict[str, Any]) -> tuple:
        payload = body.get("payload")
        if not isinstance(payload, dict) or not payload:
            return 400, {"ok": False, "reason": "payload object required"}
        try:
            reached = int(self.host.broadcast_system(payload))
        except Exception as exc:
            return 503, {"ok": False, "reason": str(exc)}
        return 200, {"ok": True, "reached": reached}


class _DashboardHandler(BaseHTTPRequestHandler):
    """One-thread-per-request handler; SSE holds its thread for the stream."""

    protocol_version = "HTTP/1.1"
    server_version = "shugonet-dashboard"

    # -- plumbing ---------------------------------------------------------------

    @property
    def dashboard(self) -> DashboardServer:
        return self.server.dashboard                     # type: ignore[attr]

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("dashboard: " + fmt, *args)

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorized(self) -> bool:
        token = self.dashboard.token
        if not token:
            return True
        header = self.headers.get("X-Shugonet-Token") or ""
        if not header:
            auth = self.headers.get("Authorization") or ""
            if auth.startswith("Bearer "):
                header = auth[len("Bearer "):]
        import hmac
        return hmac.compare_digest(str(header), token)

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0 or length > _MAX_BODY_BYTES:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return body if isinstance(body, dict) else {}

    # -- GET -----------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/status":
            self._send_json(200, self.dashboard._status_snapshot())
            return
        if path == "/api/audit":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["200"])[0])
            except (TypeError, ValueError):
                limit = 200
            body = self.dashboard._audit_tail(limit=limit)
            body["problems"] = self.dashboard._audit_problems()
            self._send_json(200, body)
            return
        if path == "/api/events":
            self._handle_events()
            return
        if path == "/api":
            self._send_json(200, {
                "service": "shugonet-dashboard",
                "endpoints": ["/api/status", "/api/audit?limit=N",
                              "/api/events (SSE)"],
            })
            return
        self._serve_static(parsed.path)

    # -- POST ------------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        path = urlparse(self.path).path.rstrip("/")
        if path not in ("/api/pair", "/api/unpair", "/api/resume",
                        "/api/broadcast"):
            self._send_json(404, {"ok": False, "reason": "unknown endpoint"})
            return
        if not self._authorized():
            self._send_json(401, {"ok": False, "reason": "token required"})
            return
        body = self._read_json_body()
        dispatch = {
            "/api/pair": self.dashboard.op_pair,
            "/api/unpair": self.dashboard.op_unpair,
            "/api/resume": self.dashboard.op_resume,
            "/api/broadcast": self.dashboard.op_broadcast,
        }
        status, result = dispatch[path](body)
        self._send_json(status, result)

    # -- SSE --------------------------------------------------------------------------

    def _handle_events(self) -> None:
        if not self._authorized():
            self._send_json(401, {"ok": False, "reason": "token required"})
            return
        dashboard = self.dashboard
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        client = dashboard.register_client()
        try:
            # Replay the ring so a fresh page has immediate context, then the
            # current status snapshot, then live frames from the broadcaster.
            for record in list(dashboard._events):
                self._write_sse("audit", record)
            self._write_sse("status", dashboard._status_snapshot())
            while not dashboard._stop.is_set():
                try:
                    event, data = client.get(timeout=_STATUS_TICK_S * 2)
                except queue.Empty:
                    self._write_sse("ping", {"ts": 0})   # keepalive
                    continue
                self._write_sse(event, data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            dashboard.unregister_client(client)

    def _write_sse(self, event: str, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, separators=(",", ":"))
        frame = f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
        self.wfile.write(frame)
        self.wfile.flush()

    # -- static SPA ----------------------------------------------------------------------

    def _serve_static(self, path: str) -> None:
        static_dir = self.dashboard.static_dir
        if not static_dir or not os.path.isdir(static_dir):
            self._send_json(200, {
                "service": "shugonet-dashboard",
                "hint": "SPA not built; run 'npm run build' in dashboard/",
            })
            return
        rel = path.lstrip("/") or "index.html"
        candidate = os.path.normpath(os.path.join(static_dir, rel))
        if not candidate.startswith(os.path.abspath(static_dir) + os.sep):
            self._send_json(403, {"ok": False, "reason": "forbidden"})
            return
        if not os.path.isfile(candidate):
            # SPA fallback: client-side routes always get the shell.
            candidate = os.path.join(static_dir, "index.html")
            if not os.path.isfile(candidate):
                self._send_json(404, {"ok": False, "reason": "not found"})
                return
            self._send_file(candidate, immutable=False)
            return
        self._send_file(candidate, immutable="/assets/" in path)

    def _send_file(self, path: str, immutable: bool) -> None:
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            self._send_json(500, {"ok": False, "reason": "read failed"})
            return
        cache = "public, max-age=31536000, immutable" if immutable \
            else "no-cache"
        self.send_response(200)
        self.send_header("Content-Type",
                         _MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
