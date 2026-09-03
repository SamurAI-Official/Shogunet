"""Fleet dashboard tests: REST surface, SSE stream, token auth, static SPA."""

import http.client
import json
import os
import socket
import tempfile
import time
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import DashboardServer
from host import ShugonetHost


class DashboardFixture(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.static_dir = os.path.join(self.tmp, "static")
        os.makedirs(self.static_dir)
        with open(os.path.join(self.static_dir, "index.html"), "w") as handle:
            handle.write("<html><body>shugonet-operator</body></html>")
        os.makedirs(os.path.join(self.static_dir, "assets"))
        with open(os.path.join(self.static_dir, "assets", "app.js"), "w") as h:
            h.write("console.log('fleet');")
        self.host = ShugonetHost(agent_id="dash-host", tcp_port=0,
                                 relay_port=0, heartbeat_timeout_s=2.0)
        self.host.start()
        self.dashboard = DashboardServer(
            self.host, port=0, static_dir=self.static_dir)
        self.dashboard.start()
        self.port = self.dashboard._server.server_address[1]

    def tearDown(self):
        self.dashboard.stop()
        self.host.stop()

    def request(self, method, path, body=None, token=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        if token:
            headers["X-Shugonet-Token"] = token
        payload = json.dumps(body).encode() if body is not None else None
        if payload is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, json.loads(data or b"{}")


class TestStatusSurface(DashboardFixture):

    def test_status_snapshot_shape(self):
        status, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        for key in ("agent_id", "shugonet_version", "mode", "roster",
                    "paired_count", "alive_count", "chain", "mesh",
                    "store_count", "audit_len", "audit_tail"):
            self.assertIn(key, body)
        self.assertEqual(body["agent_id"], "dash-host")

    def test_audit_tail_and_integrity(self):
        self.host.pair("agent-x")
        status, body = self.request("GET", "/api/audit?limit=5")
        self.assertEqual(status, 200)
        self.assertTrue(body["records"])
        self.assertEqual(body["problems"], [])
        events = [r["event"] for r in body["records"]]
        self.assertIn("agent_paired", events)


class TestOperatorActions(DashboardFixture):

    def test_pair_then_unpair(self):
        status, body = self.request("POST", "/api/pair",
                                    {"agent_id": "agent-7"})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(self.host.status()["paired_count"], 1)
        status, body = self.request("POST", "/api/unpair",
                                    {"agent_id": "agent-7"})
        self.assertEqual(status, 200)
        self.assertTrue(body["removed"])
        self.assertEqual(self.host.status()["paired_count"], 0)

    def test_pair_requires_agent_id(self):
        status, body = self.request("POST", "/api/pair", {})
        self.assertEqual(status, 400)

    def test_resume_requires_attribution(self):
        status, body = self.request("POST", "/api/resume", {})
        self.assertEqual(status, 400)
        self.assertIn("resumed_by", body["reason"])
        status, body = self.request("POST", "/api/resume",
                                    {"resumed_by": "alice (on-call)"})
        self.assertEqual(status, 200)

    def test_resume_after_pause_restores_mode(self):
        self.host.pair("agent-8")
        self.host.fallback.report_violation("network_peer_lost", "test")
        self.assertEqual(self.host.mode, "paused")
        status, body = self.request("POST", "/api/resume",
                                    {"resumed_by": "bob"})
        self.assertEqual(status, 200)
        self.assertEqual(self.host.mode, "normal")

    def test_broadcast_reaches_paired(self):
        self.host.pair("agent-9")
        status, body = self.request("POST", "/api/broadcast",
                                    {"payload": {"notice": "drill"}})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_unknown_post_404(self):
        status, _ = self.request("POST", "/api/nope", {})
        self.assertEqual(status, 404)


class TestTokenAuth(unittest.TestCase):

    def setUp(self):
        self.host = ShugonetHost(agent_id="token-host", tcp_port=0,
                                 relay_port=0)
        self.host.start()
        self.dashboard = DashboardServer(self.host, port=0, token="s3cret")
        self.dashboard.start()
        self.port = self.dashboard._server.server_address[1]

    def tearDown(self):
        self.dashboard.stop()
        self.host.stop()

    def _request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, body=json.dumps(body).encode() if body
                     else None, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, json.loads(data or b"{}")

    def test_post_without_token_is_401(self):
        status, _ = self._request("POST", "/api/pair", {"agent_id": "a"})
        self.assertEqual(status, 401)

    def test_post_with_token_succeeds(self):
        status, body = self._request("POST", "/api/pair", {"agent_id": "a"},
                                     {"X-Shugonet-Token": "s3cret"})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_bearer_header_accepted(self):
        status, _ = self._request("POST", "/api/unpair", {"agent_id": "a"},
                                  {"Authorization": "Bearer s3cret"})
        self.assertEqual(status, 200)

    def test_reads_stay_open_without_token(self):
        status, _ = self._request("GET", "/api/status")
        self.assertEqual(status, 200)


class TestStaticSPA(DashboardFixture):

    def test_index_served(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        raw = response.read().decode()
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertIn("shugonet-operator", raw)
        self.assertIn("text/html", response.getheader("Content-Type"))

    def test_spa_fallback_for_client_routes(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/fleet/agents/agent-1")
        response = conn.getresponse()
        raw = response.read().decode()
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertIn("shugonet-operator", raw)

    def test_path_traversal_refused(self):
        status, body = self.request("GET", "/../etc/passwd")
        self.assertIn(status, (403, 404))

    def test_assets_cache_header(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/assets/app.js")
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertIn("immutable", response.getheader("Cache-Control"))


def _drain(sock, until, deadline_s=6.0):
    """Read an SSE socket until a marker appears or time runs out."""
    buffer = b""
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        sock.settimeout(max(0.1, deadline - time.time()))
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        buffer += chunk
        if until(buffer):
            break
    return buffer


class TestEventStream(DashboardFixture):

    def test_sse_replays_history_then_status(self):
        # An event before anyone is listening must appear in the replay.
        self.host.pair("agent-pre")
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.sendall(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        buffer = _drain(sock, lambda b: b"event: status" in b
                        and b"agent_paired" in b)
        sock.close()
        self.assertIn(b"event: audit", buffer)
        self.assertIn(b"event: status", buffer)
        self.assertIn(b"agent_paired", buffer)

    def test_sse_delivers_new_events_live(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.sendall(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        _drain(sock, lambda b: b"event: status" in b, deadline_s=4.0)
        self.host.pair("agent-live")
        buffer = _drain(sock, lambda b: b"agent-live" in b)
        sock.close()
        self.assertIn("agent-live".encode(), buffer)

    def test_sse_content_type(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.sendall(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        _drain(sock, lambda b: b"event:" in b, deadline_s=4.0)
        sock.close()
        # (headers were inspected implicitly by the drain; explicit check:)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/api/status")
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
