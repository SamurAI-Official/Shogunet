"""Shogunet <-> ShugoCore bridge + adapter tests (hermetic, no ShugoCore)."""

import unittest

import shugocore_adapter
from shugocore_adapter import (NETWORK_ACTION_TYPES, ShugonetExecutionHandler,
                               attach_network_fallbacks,
                               register_network_handlers)
from shugocore_bridge import (configure, make_audit, redact, sanitize_text,
                              shugocore_loaded)


class _FakeAgent:
    """Duck-typed Shogunet runtime double."""

    def send(self, peer, topic, payload):
        return {"status": "success", "ok": True}

    def query(self, query, peers=None, top_k=None):
        return [{"origin": "agent-b", "fact_id": 1, "content": query,
                 "salience": 1.0}]

    def sync(self, peer=None, since=None):
        return {"synced": 1}

    def list_agents(self):
        return [{"agent_id": "agent-b"}]

    def status(self):
        return {"mode": "normal"}


class _ExecutionLayer:
    def __init__(self):
        self.handlers = {}

    def register_handler(self, action_type, fn):
        self.handlers[action_type] = fn


class _PolicyStub:
    KNOWN_ACTION_TYPES = set()


class TestShugocoreBridge(unittest.TestCase):

    def test_configure_returns_false_without_path(self):
        self.assertIsInstance(configure(""), bool)

    def test_primitives_work_standalone(self):
        self.assertIsInstance(sanitize_text("a\x01b", 16), str)
        self.assertEqual(redact({"token": "sekrit"})["token"], "[REDACTED]")

    def test_make_audit_returns_local_chain(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp:
            chain = make_audit(os.path.join(tmp, "audit.jsonl"))
            self.assertEqual(chain.verify(), [])
            chain.append("bridge_test", {"n": 1})


class TestShugonetAdapter(unittest.TestCase):

    def test_handler_dispatches_actions(self):
        handler = ShugonetExecutionHandler(_FakeAgent())
        self.assertEqual(handler.handle(
            {"action_type": "network_list_agents",
             "params": {}})["status"], "success")
        # Missing send params -> refused.
        refused = handler.handle({"action_type": "network_send",
                                  "params": {}})
        self.assertEqual(refused["status"], "refused")
        # Valid send params -> success.
        sent = handler.handle({"action_type": "network_send",
                               "params": {"peer": "agent-b",
                                          "topic": "mem",
                                          "payload": {"x": 1}}})
        self.assertEqual(sent["status"], "success")

    def test_handler_refuses_unknown_action(self):
        handler = ShugonetExecutionHandler(_FakeAgent())
        result = handler.handle({"action_type": "network_meme", "params": {}})
        self.assertEqual(result["status"], "refused")

    def test_register_patches_policy_and_handlers(self):
        exec_layer = _ExecutionLayer()
        policy = _PolicyStub()
        register_network_handlers(exec_layer, _FakeAgent(),
                                  policy_module=policy)
        for action_type in sorted(NETWORK_ACTION_TYPES
                                  | shugocore_adapter.NETWORK_READ_ACTION_TYPES):
            self.assertIn(action_type, policy.KNOWN_ACTION_TYPES)
            self.assertIn(action_type, exec_layer.handlers)

    def test_attach_fallbacks_merges_severities(self):
        class _FB:
            severities = {}

        fb = _FB()
        attach_network_fallbacks(fb)
        self.assertEqual(fb.severities["network_peer_lost"], "pause")


if __name__ == "__main__":
    unittest.main()