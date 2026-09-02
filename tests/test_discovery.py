"""Shogunet discovery tests: multicast beacons, registry liveness, filtering."""

import time
import unittest

from agent_registry import AgentRegistry
from discovery import UDPMulticastDiscovery


class TestDiscovery(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()
        self.seen = []
        self.d1 = UDPMulticastDiscovery(
            "agent-a", registry=self.registry, interval_s=0.2,
            manifest={"role": "scout", "realm": "phys"},
            on_peer=lambda agent, manifest: self.seen.append((agent, manifest)))
        self.d2 = UDPMulticastDiscovery(
            "agent-b", registry=self.registry, interval_s=0.2,
            manifest={"role": "base"})
        self.d1.start()
        self.d2.start()

    def tearDown(self):
        self.d1.stop()
        self.d2.stop()

    def _wait_for_peer(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if "agent-a" in self.d2.peers():
                return True
            time.sleep(0.05)
        return False

    def test_peer_learned_from_beacons(self):
        self.assertTrue(self._wait_for_peer())
        peers = self.d2.peers()
        self.assertIn("agent-a", peers)
        self.assertEqual(peers["agent-a"]["manifest"].get("role"), "scout")

    def test_on_peer_callback_fires(self):
        self.assertTrue(self._wait_for_peer())
        # d1's on_peer fires for the peers IT observes (agent-b), and
        # d2's for agent-a.
        self.assertTrue(any(agent == "agent-b" for agent, _ in self.seen))

    def test_registry_liveness_fed(self):
        self.assertTrue(self._wait_for_peer())
        self.assertTrue(self.registry.is_paired("agent-a") is False)
        # heartbeat() on an unpaired agent returns False -- pairing stays
        # the operator's job; discovery only refreshes known agents.
        self.registry.pair("agent-a")
        self.assertTrue(self.registry.heartbeat("agent-a"))

    def test_own_beacons_ignored(self):
        self.assertTrue(self._wait_for_peer())
        self.assertNotIn("agent-b", self.d2.peers())
        self.assertNotIn("agent-a", self.d1.peers())

    def test_bidirectional_peering(self):
        self.assertTrue(self._wait_for_peer())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if "agent-b" in self.d1.peers():
                break
            time.sleep(0.05)
        self.assertIn("agent-b", self.d1.peers())


if __name__ == "__main__":
    unittest.main()
