"""
Tests for ShugonetHost + ShugonetAgentRuntime integration.

Verifies the server-host path for multiple ShugoCore agents:
connected, refused, convergence, peer-lost, shutdown.
"""

import threading
import time
import unittest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host import ShugonetHost
from agent_runtime import ShugonetAgentRuntime


class TestHostIntegration(unittest.TestCase):

    def _make_host(self):
        return ShugonetHost(agent_id="test-host", tcp_port=0, relay_port=0,
                            heartbeat_timeout_s=2.0)

    def _make_runtime(self, agent_id, host):
        return ShugonetAgentRuntime(
            agent_id=agent_id,
            host_tcp_host="127.0.0.1",
            host_tcp_port=host.tcp_port,
            host_agent_id=host.agent_id,
            host_relay_url=host.relay_url,
            realm="sim")

    def test_host_and_three_agents_connected(self):
        host = self._make_host()
        host.start()
        try:
            agents = []
            received = {f"agent-{i}": [] for i in range(3)}
            barrier = threading.Barrier(3)

            def make_callback(aid):
                def cb(sender, msg):
                    received[aid].append((sender, msg))
                return cb

            for i in range(3):
                aid = f"agent-{i}"
                host.pair(aid)
                rt = self._make_runtime(aid, host)
                rt.on_message = make_callback(aid)
                rt.connect_to_host()
                agents.append(rt)

            deadline = time.time() + 5.0
            while time.time() < deadline:
                st = host.status()
                if st["alive_count"] >= 3:
                    break
                time.sleep(0.1)

            st = host.status()
            self.assertEqual(st["paired_count"], 3)
            self.assertGreaterEqual(st["alive_count"], 3)

            agents[0].send("agent-1", "/test", {"msg": "hello"})
            agents[1].send("agent-2", "/test", {"msg": "world"})

            deadline = time.time() + 3.0
            while time.time() < deadline:
                if len(received["agent-1"]) > 0 and len(received["agent-2"]) > 0:
                    break
                time.sleep(0.05)

            self.assertTrue(len(received["agent-1"]) > 0)
            self.assertTrue(len(received["agent-2"]) > 0)

            for sender, msg in received["agent-0"]:
                self.assertNotEqual(msg.get("msg"), "hello")

            for rt in agents:
                rt.stop()
        finally:
            host.stop()

    def test_unpaired_agent_refused(self):
        host = self._make_host()
        host.start()
        try:
            rt = ShugonetAgentRuntime(
                agent_id="unpaired",
                host_tcp_host="127.0.0.1",
                host_tcp_port=host.tcp_port,
                host_agent_id=host.agent_id)
            rt.connect_to_host()
            # Wait for the handshake to complete (or fail)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if not rt.tcp.has_active_connection():
                    break
                time.sleep(0.05)
            # The unpaired agent should NOT have an active connection
            self.assertFalse(rt.tcp.has_active_connection(),
                             "unpaired agent should be refused at TCP admission")
        finally:
            rt.stop()
            host.stop()

    def test_agent_can_send_addressed_message_via_host(self):
        """Core routing test: agent A sends to agent B via the host."""
        host = self._make_host()
        host.start()
        try:
            host.pair("agent-a")
            host.pair("agent-b")
            a = self._make_runtime("agent-a", host)
            b = self._make_runtime("agent-b", host)
            received = []
            b.on_message = lambda sender, msg: received.append((sender, msg))
            a.connect_to_host()
            b.connect_to_host()

            a.send("agent-b", "/test/greeting", {"hello": "world"})

            deadline = time.time() + 5.0
            while time.time() < deadline:
                host.chain.poll(0.02)
                a.chain.poll(0.02)
                b.chain.poll(0.02)
                # Wait for the task_request (not the announce which is broadcast)
                if any(m.get("msg_type") == "task_request" for _, m in received):
                    break
                time.sleep(0.05)

            task_msgs = [(s, m) for s, m in received if m.get("msg_type") == "task_request"]
            self.assertTrue(task_msgs, "agent-b should receive the task_request")
            self.assertEqual(task_msgs[0][0], "agent-a")
            self.assertIn("hello", task_msgs[0][1]["payload"])
            self.assertEqual(task_msgs[0][1]["payload"]["hello"], "world")

            a.stop()
            b.stop()
        finally:
            host.stop()

    def test_peer_lost_latch(self):
        host = self._make_host()
        host.start()
        try:
            host.pair("agent-a")
            rt = self._make_runtime("agent-a", host)
            rt.connect_to_host()

            deadline = time.time() + 3.0
            while time.time() < deadline:
                st = host.status()
                if st["alive_count"] >= 1:
                    break
                time.sleep(0.1)

            rt.stop()

            deadline = time.time() + 5.0
            while time.time() < deadline:
                st = host.status()
                if st["fallback"]["mode"] == "paused":
                    break
                time.sleep(0.1)

            st = host.status()
            self.assertEqual(st["fallback"]["mode"], "paused")
            self.assertIn("network_peer_lost", st["fallback"]["violations"])

            host.resume("test-operator")
            self.assertEqual(host.status()["fallback"]["mode"], "normal")
        finally:
            host.stop()

    def test_clean_shutdown(self):
        host = self._make_host()
        host.start()
        agents = []
        try:
            for i in range(3):
                aid = f"agent-{i}"
                host.pair(aid)
                rt = self._make_runtime(aid, host)
                rt.connect_to_host()
                agents.append(rt)
            time.sleep(0.5)
        finally:
            host.stop()
            for rt in agents:
                rt.stop()

        self.assertFalse(host._running)


if __name__ == "__main__":
    unittest.main()
