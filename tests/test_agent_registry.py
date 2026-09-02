"""Shogunet agent registry tests: pairing consent, TTLs, liveness, topic ACLs."""

import time
import unittest

from agent_registry import AgentRegistry, parse_shugonet_topic


class _AuditSpy:
    def __init__(self):
        self.events = []

    def append(self, event_type, payload):
        self.events.append((event_type, dict(payload)))


class TestTopicParsing(unittest.TestCase):

    def test_valid_topics(self):
        self.assertEqual(parse_shugonet_topic("/shugunet/agent-a/memory"),
                         ("agent-a", "memory"))
        self.assertEqual(parse_shugonet_topic("shugunet/agent-a/tasks"),
                         ("agent-a", "tasks"))

    def test_invalid_topics(self):
        for topic in ("", "/", "/other/agent-a/memory",
                      "/shugunet/agent-a/memory/extra", "/shugunet//memory"):
            self.assertIsNone(parse_shugonet_topic(topic))


class TestPairing(unittest.TestCase):

    def setUp(self):
        self.spy = _AuditSpy()
        self.registry = AgentRegistry(audit=self.spy)

    def test_pair_requires_consent_then_admits(self):
        self.assertFalse(self.registry.is_paired("agent-a"))
        self.registry.pair("agent-a", manifest={"realm": "sim"})
        self.assertTrue(self.registry.is_paired("agent-a"))
        self.assertEqual(self.registry.realm("agent-a"), "sim")
        self.assertEqual(self.spy.events[0][0], "agent_paired")

    def test_pair_requires_id(self):
        with self.assertRaises(ValueError):
            self.registry.pair("   ")

    def test_unpair_revokes(self):
        self.registry.pair("agent-a")
        self.assertTrue(self.registry.unpair("agent-a"))
        self.assertFalse(self.registry.is_paired("agent-a"))
        self.assertFalse(self.registry.unpair("agent-a"))
        self.assertEqual(self.spy.events[-1][0], "agent_unpaired")

    def test_ttl_expiry(self):
        registry = AgentRegistry(audit=self.spy, pairing_ttl_hours=0.01)
        registry.pair("agent-a")
        with registry._lock:
            registry._paired["agent-a"]["expires_at"] = time.time() - 1.0
        self.assertFalse(registry.is_paired("agent-a"))
        self.assertEqual(self.spy.events[-1][0], "agent_pairing_expired")


class TestLiveness(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry(heartbeat_timeout_s=30.0)

    def test_heartbeat_requires_pairing(self):
        self.assertFalse(self.registry.heartbeat("ghost"))
        self.registry.pair("agent-a")
        self.assertTrue(self.registry.heartbeat("agent-a"))
        self.assertTrue(self.registry.alive("agent-a"))

    def test_stale_agent_not_alive(self):
        self.registry.pair("agent-a")
        self.registry.heartbeat("agent-a")
        with self.registry._lock:
            key = "agent-a"
            self.registry._last_heartbeat[key] = (
                time.monotonic() - self.registry.heartbeat_timeout_s - 1.0)
        self.assertFalse(self.registry.alive("agent-a"))
        self.assertTrue(self.registry.is_paired("agent-a"))  # stale != unpaired

    def test_roster(self):
        self.registry.pair("agent-a", manifest={"role": "scout"})
        self.registry.pair("agent-b")
        roster = self.registry.list_agents()
        self.assertEqual({entry["agent_id"] for entry in roster},
                         {"agent-a", "agent-b"})
        self.assertEqual(self.registry.manifest("agent-a"), {"role": "scout"})


class TestTopicACL(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()
        self.registry.pair("agent-a")
        self.registry.pair("agent-b")

    def test_publish_own_namespace_only(self):
        self.assertTrue(self.registry.can_publish(
            "agent-a", "/shugunet/agent-a/memory"))
        self.assertFalse(self.registry.can_publish(
            "agent-a", "/shugunet/agent-b/memory"))     # cross-namespace
        self.assertFalse(self.registry.can_publish(
            "agent-a", "/shugocore/agent-a/memory"))    # wrong namespace
        self.assertFalse(self.registry.can_publish(
            "ghost", "/shugunet/ghost/memory"))         # unpaired

    def test_accept_sender_namespace_only(self):
        self.assertTrue(self.registry.can_accept(
            "agent-a", "/shugunet/agent-a/memory"))
        self.assertFalse(self.registry.can_accept(
            "agent-a", "/shugunet/agent-b/memory"))
        self.assertFalse(self.registry.can_accept(
            "ghost", "/shugunet/ghost/memory"))


if __name__ == "__main__":
    unittest.main()
