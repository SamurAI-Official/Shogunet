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
    """Stand-in for ShugoCore's ``policy`` module.

    Carries the two sets ``ExecutionLayer.register_handler`` actually
    validates against. Instance attributes (not class attributes) so an
    in-place merge in one test cannot leak into the next.
    """

    def __init__(self):
        self.KNOWN_ACTION_TYPES = set()
        self.NETWORK_ACTION_TYPES = set()
        self.NETWORK_READ_ACTION_TYPES = set()


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

    def test_register_patches_the_sets_execution_layer_validates(self):
        # ExecutionLayer.register_handler validates against NETWORK_ACTION_TYPES
        # / NETWORK_READ_ACTION_TYPES, not KNOWN_ACTION_TYPES. Patching only the
        # latter is what made real registration raise ValueError.
        policy = _PolicyStub()
        register_network_handlers(_ExecutionLayer(), _FakeAgent(),
                                  policy_module=policy)
        self.assertTrue(NETWORK_ACTION_TYPES.issubset(
            policy.NETWORK_ACTION_TYPES))
        self.assertTrue(
            shugocore_adapter.NETWORK_READ_ACTION_TYPES.issubset(
                policy.NETWORK_READ_ACTION_TYPES))

    def test_register_mutates_policy_sets_in_place(self):
        # execution_layer holds a reference to these objects; rebinding would
        # leave it validating a stale set forever.
        policy = _PolicyStub()
        before = policy.NETWORK_ACTION_TYPES
        register_network_handlers(_ExecutionLayer(), _FakeAgent(),
                                  policy_module=policy)
        self.assertIs(policy.NETWORK_ACTION_TYPES, before)
        self.assertIn("network_send", before)

    def test_register_keeps_the_types_a_strict_layer_accepts(self):
        # A layer that refuses the spatial/NRR types must still get the five
        # it does accept, rather than losing the whole batch to one refusal.
        class _StrictLayer:
            def __init__(self, allowed):
                self.allowed = set(allowed)
                self.handlers = {}

            def register_handler(self, action_type, fn):
                if action_type not in self.allowed:
                    raise ValueError("not allowed: " + action_type)
                self.handlers[action_type] = fn

        layer = _StrictLayer({"network_send", "network_query", "network_sync",
                              "network_list_agents", "network_status"})
        registered = register_network_handlers(layer, _FakeAgent(),
                                               policy_module=_PolicyStub())
        self.assertEqual(sorted(registered),
                         ["network_list_agents", "network_query", "network_send",
                          "network_status", "network_sync"])

    def test_register_raises_only_when_everything_is_rejected(self):
        class _ClosedLayer:
            def register_handler(self, action_type, fn):
                raise ValueError("closed: " + action_type)

        with self.assertRaises(RuntimeError):
            register_network_handlers(_ClosedLayer(), _FakeAgent(),
                                      policy_module=_PolicyStub())

    def test_register_returns_registered_types(self):
        registered = register_network_handlers(_ExecutionLayer(), _FakeAgent(),
                                               policy_module=_PolicyStub())
        self.assertEqual(sorted(registered), sorted(
            NETWORK_ACTION_TYPES | shugocore_adapter.NETWORK_READ_ACTION_TYPES))

    def test_attach_fallbacks_merges_severities(self):
        class _FB:
            severities = {}

        fb = _FB()
        attach_network_fallbacks(fb)
        self.assertEqual(fb.severities["network_peer_lost"], "pause")


class _SpatialAgent(_FakeAgent):
    """Runtime double with the ShugoCore 1.30 spatial + NRR surface."""

    def __init__(self):
        self.rendered = []
        self.scenes = []

    def publish_observation(self, entity_id, x, y, z, confidence=1.0,
                            label="", frame_id="world"):
        return {"status": "success", "entity_id": entity_id}

    def query_nearby(self, x, y, z, radius):
        return [{"entity_id": "robot-1", "x": x, "y": y}]

    def get_fleet_map(self):
        return {"entities": {}, "agents": {}}

    def render_frame(self, peer, descriptor):
        self.rendered.append((peer, descriptor))
        return {"status": "success", "request_id": "nrr-1"}

    def query_scene(self, peer, motion_request):
        self.scenes.append((peer, motion_request))
        return {"status": "success", "request_id": "nrr-2"}


class TestShugonetSpatialAndNRRActions(unittest.TestCase):

    def setUp(self):
        self.agent = _SpatialAgent()
        self.handler = ShugonetExecutionHandler(self.agent)

    def test_spatial_actions(self):
        pub = self.handler.handle({
            "action_type": "network_spatial_observe",
            "params": {"entity_id": "robot-1", "x": 1, "y": 2}})
        self.assertEqual(pub["status"], "success")
        missing = self.handler.handle({"action_type": "network_spatial_observe",
                                       "params": {"x": 1}})
        self.assertEqual(missing["status"], "refused")
        self.assertEqual(self.handler.handle({
            "action_type": "network_spatial_query",
            "params": {"x": 1, "y": 2, "radius": 5}})["status"], "success")
        self.assertEqual(self.handler.handle({
            "action_type": "network_fleet_map", "params": {}})["status"],
            "success")

    def test_nrr_actions_require_peer_and_payload(self):
        ok = self.handler.handle({"action_type": "network_nrr_render",
                                  "params": {"peer": "gpu-1",
                                             "descriptor": {"frame_id": "f1"}}})
        self.assertEqual(ok["status"], "success")
        self.assertEqual(self.agent.rendered[0][0], "gpu-1")
        for action in ("network_nrr_render", "network_nrr_scene"):
            self.assertEqual(self.handler.handle(
                {"action_type": action, "params": {}})["status"], "refused")

    def test_nrr_scene_action(self):
        res = self.handler.handle({"action_type": "network_nrr_scene",
                                   "params": {"peer": "gpu-1",
                                              "motion_request": {"frame_id": "f"}}})
        self.assertEqual(res["status"], "success")
        self.assertEqual(self.agent.scenes[0][0], "gpu-1")

    def test_capabilities_absent_runtime_refuses_cleanly(self):
        # An older runtime without these methods must be refused, not crash.
        bare = ShugonetExecutionHandler(_FakeAgent())
        for action, params in (("network_nrr_render",
                                {"peer": "gpu-1", "descriptor": {}}),
                               ("network_nrr_scene",
                                {"peer": "gpu-1", "motion_request": {}}),
                               ("network_spatial_query",
                                {"x": 1, "y": 1, "radius": 1}),
                               ("network_fleet_map", {}),
                               ("network_spatial_observe",
                                {"entity_id": "r", "x": 1, "y": 1})):
            res = bare.handle({"action_type": action, "params": params})
            self.assertEqual(res["status"], "refused", action)

    def test_every_registered_type_is_handled(self):
        for action in (NETWORK_ACTION_TYPES
                       | shugocore_adapter.NETWORK_READ_ACTION_TYPES):
            result = self.handler.handle({"action_type": action,
                                          "params": {}})
            self.assertIn(result["status"], ("success", "refused"), action)
            self.assertNotIn("unknown network action", result.get("reason",
                                                                   ""), action)


if __name__ == "__main__":
    unittest.main()