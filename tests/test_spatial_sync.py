"""SpatialMemoryNode integration tests: cross-agent observation sharing,
queries, and fleet map consolidation over a loopback bus."""

import time
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial import SpatialIndex, SpatialObservation, FrameTransform
from spatial_sync import SpatialMemoryNode
from transports import LoopbackBus, LoopbackTransport
from transport_fallback import TransportChain


class _PollMixin:
    """Helper to pump chains so loopback deliveries drain."""

    @staticmethod
    def pump(*chains, rounds=30):
        for _ in range(rounds):
            for c in chains:
                c.poll(0.01)
            time.sleep(0.005)


class TestSpatialMemoryNodePublish(unittest.TestCase):

    def setUp(self):
        self.bus = LoopbackBus("spatial-test")
        self.a = LoopbackTransport("agent-a", self.bus)
        self.b = LoopbackTransport("agent-b", self.bus)
        self.chain_a = TransportChain("agent-a", [self.a])
        self.chain_b = TransportChain("agent-b", [self.b])
        self.idx_a = SpatialIndex()
        self.idx_b = SpatialIndex()
        self.node_a = SpatialMemoryNode("agent-a", self.chain_a, self.idx_a)
        self.node_b = SpatialMemoryNode("agent-b", self.chain_b, self.idx_b)

    def _obs(self, eid, aid, x, y, z, c=0.9):
        return SpatialObservation(entity_id=eid, agent_id=aid,
                                  x=x, y=y, z=z, confidence=c,
                                  timestamp=time.time())

    def test_publish_reaches_peer(self):
        self.node_a.publish_observation(self._obs("robot-1", "agent-a",
                                                   10, 20, 0))
        _PollMixin.pump(self.chain_a, self.chain_b)
        received = self.idx_b.all_observations()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].entity_id, "robot-1")

    def test_publish_inserts_local(self):
        self.node_a.publish_observation(self._obs("robot-1", "agent-a",
                                                   10, 20, 0))
        self.assertEqual(len(self.idx_a.all_observations()), 1)

    def test_publish_position(self):
        self.node_a.publish_position(5, 10, 0)
        _PollMixin.pump(self.chain_a, self.chain_b)
        self.assertEqual(len(self.idx_a.all_observations()), 1)
        self.assertEqual(len(self.idx_b.all_observations()), 1)

    def test_publish_transform_reaches_peer(self):
        t = FrameTransform(from_frame="agent-a-local", to_frame="world",
                           tx=10, ty=0, tz=0)
        self.node_a.publish_transform(t)
        _PollMixin.pump(self.chain_a, self.chain_b)
        key = "agent-a-local\u2192world"
        self.assertIn(key, self.idx_a._transforms)
        self.assertIn(key, self.idx_b._transforms)
class TestSpatialMemoryNodeQuery(unittest.TestCase):

    def setUp(self):
        self.bus = LoopbackBus("spatial-query")
        self.a = LoopbackTransport("agent-a", self.bus)
        self.b = LoopbackTransport("agent-b", self.bus)
        self.chain_a = TransportChain("agent-a", [self.a])
        self.chain_b = TransportChain("agent-b", [self.b])
        self.idx_a = SpatialIndex()
        self.idx_b = SpatialIndex()
        self.node_a = SpatialMemoryNode("agent-a", self.chain_a, self.idx_a)
        self.node_b = SpatialMemoryNode("agent-b", self.chain_b, self.idx_b)

    def _obs(self, eid, aid, x, y, z, c=0.9):
        return SpatialObservation(entity_id=eid, agent_id=aid,
                                  x=x, y=y, z=z, confidence=c,
                                  timestamp=time.time())

    def test_query_nearby_local(self):
        self.node_a.publish_observation(self._obs("r1", "agent-a", 5, 5, 0))
        results = self.node_a.query_nearby(5, 5, 0, 10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_id, "r1")

    def test_query_sphere_networked(self):
        self.node_b.publish_observation(self._obs("r1", "agent-b", 5, 5, 0))
        _PollMixin.pump(self.chain_a, self.chain_b)
        results = self.node_a.query_sphere(
            5, 5, 0, 10,
            poller=lambda: _PollMixin.pump(self.chain_a, self.chain_b, rounds=5))
        self.assertTrue(results, "expected results from networked query")
        self.assertEqual(results[0].entity_id, "r1")

    def test_query_sphere_no_results(self):
        self.node_b.publish_observation(self._obs("r1", "agent-b", 100, 100, 0))
        _PollMixin.pump(self.chain_a, self.chain_b)
        results = self.node_a.query_sphere(
            0, 0, 0, 1,
            poller=lambda: _PollMixin.pump(self.chain_a, self.chain_b, rounds=5))
        self.assertEqual(results, [])

    def test_auto_answer_ignores_self(self):
        # Agent-a has local data but must NOT answer its own query.
        # Agent-b is empty, so its (empty) response still completes the query.
        self.idx_a.insert(self._obs("r1", "agent-a", 5, 5, 0))
        results = self.node_a.query_sphere(
            5, 5, 0, 10, timeout_s=0.4,
            poller=lambda: _PollMixin.pump(self.chain_a, self.chain_b, rounds=5))
        self.assertEqual(results, [])


class TestSpatialMemoryNodeFleetMap(unittest.TestCase):

    def setUp(self):
        self.bus = LoopbackBus("spatial-map")
        self.a = LoopbackTransport("agent-a", self.bus)
        self.b = LoopbackTransport("agent-b", self.bus)
        self.chain_a = TransportChain("agent-a", [self.a])
        self.chain_b = TransportChain("agent-b", [self.b])
        self.idx_a = SpatialIndex()
        self.idx_b = SpatialIndex()
        self.node_a = SpatialMemoryNode("agent-a", self.chain_a, self.idx_a)
        self.node_b = SpatialMemoryNode("agent-b", self.chain_b, self.idx_b)

    def _obs(self, eid, aid, x, y, z, c=0.9):
        return SpatialObservation(entity_id=eid, agent_id=aid,
                                  x=x, y=y, z=z, confidence=c,
                                  timestamp=time.time())

    def test_get_fleet_map_consolidates(self):
        self.node_a.publish_observation(self._obs("r1", "agent-a", 5, 5, 0))
        self.node_a.publish_observation(self._obs("r2", "agent-a", 10, 10, 0))
        _PollMixin.pump(self.chain_a, self.chain_b)
        fm = self.node_b.get_fleet_map()
        self.assertIn("r1", fm["entities"])
        self.assertIn("r2", fm["entities"])
        self.assertEqual(fm["observation_count"], 2)

    def test_get_agent_position(self):
        self.node_a.publish_position(5, 10, 0)
        _PollMixin.pump(self.chain_a, self.chain_b)
        pos = self.node_b.get_agent_position("agent-a")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.entity_id, "agent-a")
        self.assertEqual(pos.x, 5.0)
        self.assertEqual(pos.y, 10.0)

    def test_get_agent_position_unknown(self):
        self.assertIsNone(self.node_a.get_agent_position("unknown-agent"))

    def test_send_merge_reaches_peers(self):
        self.node_a.publish_observation(self._obs("r1", "agent-a", 5, 5, 0))
        _PollMixin.pump(self.chain_a, self.chain_b)
        sent = self.node_a.send_merge()
        self.assertTrue(sent)
        _PollMixin.pump(self.chain_a, self.chain_b)
        # The merged fused snapshot lands on agent-b (original + fused copy)
        entities = {o.entity_id for o in self.idx_b.all_observations()}
        self.assertIn("r1", entities)
        self.assertGreaterEqual(self.node_a.stats()["merges_sent"], 1)

    def test_duplicate_observation_dedup(self):
        """The same observation delivered twice must land only once."""
        from protocol import Envelope
        import protocol as _protocol
        obs = self._obs("r1", "agent-a", 5, 5, 0)
        env = Envelope(msg_id=_protocol.new_msg_id(),
                       msg_type="spatial_observation",
                       sender="agent-a", recipient="*",
                       topic="/shugunet/agent-a/spatial",
                       payload=obs.to_dict())
        self.node_b._handle_observation(env)
        self.node_b._handle_observation(env)   # same payload again
        self.assertEqual(len(self.idx_b.all_observations()), 1)
        self.assertEqual(self.idx_b.stats()["duplicates_skipped"], 1)


class TestSpatialMemoryNodeProtocol(unittest.TestCase):

    def test_message_types_registered(self):
        import protocol
        self.assertEqual(protocol.MESSAGE_TYPES["spatial_observation"], 14)
        self.assertEqual(protocol.MESSAGE_TYPES["spatial_query"], 15)
        self.assertEqual(protocol.MESSAGE_TYPES["spatial_response"], 16)
        self.assertEqual(protocol.MESSAGE_TYPES["spatial_merge"], 17)
        self.assertEqual(protocol.MESSAGE_TYPES["coordinate_frame"], 18)

    def test_type_classes(self):
        import protocol
        self.assertEqual(protocol.TYPE_CLASS["spatial_observation"],
                         protocol.CLASS_MEMORY)
        self.assertEqual(protocol.TYPE_CLASS["spatial_query"],
                         protocol.CLASS_TASK)
        self.assertEqual(protocol.TYPE_CLASS["coordinate_frame"],
                         protocol.CLASS_CONTROL)


if __name__ == "__main__":
    unittest.main()