"""Shogunet network fallback + policy tests."""

import unittest

from fallbacks import (NETWORK_FALLBACK_SEVERITIES, NetworkFallbackController,
                       NetworkFallbackHalt)
from policy import (KNOWN_ACTION_TYPES, NETWORK_ACTION_TYPES,
                    NETWORK_READ_ACTION_TYPES, network_topic)


class _Governor:
    """Duck-typed ShugoCore governor double recording latched states."""

    def __init__(self):
        self.state = "normal"
        self.calls = []

    def pause(self, reason):
        self.state = "paused"
        self.calls.append(("pause", reason))

    def safe_state(self, reason):
        self.state = "safe_state"
        self.calls.append(("safe_state", reason))

    def halt(self, reason):
        self.state = "halted"
        self.calls.append(("halt", reason))

    def resume(self, resumed_by=""):
        self.state = "normal"
        self.calls.append(("resume", resumed_by))


class _AuditSpy:
    def __init__(self):
        self.events = []

    def append(self, event_type, payload):
        self.events.append((event_type, dict(payload)))


class TestPolicy(unittest.TestCase):

    def test_action_type_sets(self):
        self.assertIn("network_send", NETWORK_ACTION_TYPES)
        self.assertIn("network_query", NETWORK_ACTION_TYPES)
        self.assertIn("network_sync", NETWORK_ACTION_TYPES)
        self.assertIn("network_list_agents", NETWORK_READ_ACTION_TYPES)
        self.assertTrue(NETWORK_ACTION_TYPES <= KNOWN_ACTION_TYPES)
        self.assertTrue(NETWORK_READ_ACTION_TYPES <= KNOWN_ACTION_TYPES)

    def test_network_topic_namespace(self):
        self.assertEqual(network_topic("agent-a", "memory"),
                         "/shugunet/agent-a/memory")


class TestNetworkFallbackController(unittest.TestCase):

    def setUp(self):
        self.governor = _Governor()
        self.audit = _AuditSpy()
        self.controller = NetworkFallbackController(self.governor,
                                                    audit=self.audit)

    def test_peer_lost_pauses(self):
        self.controller.report_violation("network_peer_lost", "agent-b gone")
        self.assertEqual(self.governor.state, "paused")
        self.assertEqual(self.controller.mode, "paused")
        self.assertTrue(any(e[0] == "network_fallback_trigger"
                            for e in self.audit.events))

    def test_transport_exhausted_pauses(self):
        self.controller.report_violation("network_transport_exhausted",
                                         "all transports open")
        self.assertEqual(self.governor.state, "paused")

    def test_conflict_storm_safe_state_after_threshold(self):
        controller = NetworkFallbackController(
            self.governor, audit=self.audit, conflict_storm_threshold=5)
        for _ in range(4):
            controller.report_violation("memory_sync_conflict_storm", "x")
        self.assertEqual(self.governor.state, "normal")
        controller.report_violation("memory_sync_conflict_storm", "x")
        self.assertEqual(self.governor.state, "safe_state")

    def test_audit_chain_broken_halts(self):
        with self.assertRaises(NetworkFallbackHalt):
            self.controller.report_violation("audit_chain_broken", "tamper")
        self.assertEqual(self.governor.state, "halted")
        self.assertEqual(self.controller.mode, "halted")

    def test_resume_requires_attribution(self):
        self.controller.report_violation("network_peer_lost", "gone")
        self.controller.resume(resumed_by="operator")
        self.assertEqual(self.governor.state, "normal")
        self.assertEqual(self.controller.mode, "normal")
        self.assertTrue(any(e[0] == "network_fallback_resume"
                            for e in self.audit.events))

    def test_severity_map_exported(self):
        self.assertEqual(NETWORK_FALLBACK_SEVERITIES["network_peer_lost"],
                         "pause")
        self.assertEqual(NETWORK_FALLBACK_SEVERITIES["audit_chain_broken"],
                         "halt")


if __name__ == "__main__":
    unittest.main()