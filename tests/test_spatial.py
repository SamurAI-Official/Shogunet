"""Spatial data structure tests: SpatialObservation, SpatialIndex, fusion."""
import time
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial import SpatialObservation, SpatialIndex, FrameTransform, MIN_CONFIDENCE


class TestSpatialObservation(unittest.TestCase):

    def test_minimal_observation(self):
        obs = SpatialObservation(entity_id="robot-1", agent_id="agent-a",
                                 x=1.0, y=2.0, z=3.0, confidence=0.9, timestamp=1000.0)
        self.assertEqual(obs.entity_id, "robot-1")
        self.assertEqual(obs.x, 1.0)
        self.assertEqual(obs.confidence, 0.9)
        self.assertEqual(obs.frame_id, "world")

    def test_entity_id_required(self):
        with self.assertRaises(ValueError):
            SpatialObservation(entity_id="", agent_id="a", x=0, y=0, z=0,
                               confidence=0.5, timestamp=0)

    def test_confidence_clamped(self):
        obs = SpatialObservation(entity_id="e", agent_id="a", x=0, y=0, z=0,
                                 confidence=2.5, timestamp=0)
        self.assertEqual(obs.confidence, 1.0)
        obs2 = SpatialObservation(entity_id="e", agent_id="a", x=0, y=0, z=0,
                                  confidence=-1.0, timestamp=0)
        self.assertEqual(obs2.confidence, MIN_CONFIDENCE)

    def test_round_trip_dict(self):
        obs = SpatialObservation(entity_id="robot-1", agent_id="agent-a",
                                 x=5.0, y=10.0, z=0.0, confidence=0.9,
                                 timestamp=2000.0, label="robot",
                                 metadata={"color": "red"})
        d = obs.to_dict()
        obs2 = SpatialObservation.from_dict(d)
        self.assertEqual(obs.entity_id, obs2.entity_id)
        self.assertEqual(obs.x, obs2.x)

    def test_age(self):
        now = time.time()
        obs = SpatialObservation(entity_id="e", agent_id="a", x=0, y=0, z=0,
                                 confidence=0.5, timestamp=now - 10.0)
        self.assertAlmostEqual(obs.age(now), 10.0, places=1)

    def test_effective_confidence_no_decay(self):
        now = time.time()
        obs = SpatialObservation(entity_id="e", agent_id="a", x=0, y=0, z=0,
                                 confidence=0.8, timestamp=now - 10.0)
        self.assertAlmostEqual(obs.effective_confidence(now), 0.8)

    def test_effective_confidence_decayed(self):
        now = time.time()
        obs = SpatialObservation(entity_id="e", agent_id="a", x=0, y=0, z=0,
                                 confidence=0.8, timestamp=now - 90.0)
        # 90s old with 60s decay window: factor = 1 - (90-60)/60 = 0.5
        ec = obs.effective_confidence(now, decay_s=60.0)
        self.assertAlmostEqual(ec, 0.4, places=2)

    def test_effective_confidence_fully_expired(self):
        now = time.time()
        obs = SpatialObservation(entity_id="e", agent_id="a", x=0, y=0, z=0,
                                 confidence=0.8, timestamp=now - 200.0)
        ec = obs.effective_confidence(now, decay_s=60.0)
        self.assertEqual(ec, 0.0)

    def test_effective_confidence_expired(self):
        now = time.time()
        obs = SpatialObservation(entity_id="e", agent_id="a", x=0, y=0, z=0,
                                 confidence=0.5, timestamp=now - 200.0)
        ec = obs.effective_confidence(now, decay_s=60.0)
        self.assertAlmostEqual(ec, 0.0)


class TestSpatialIndex(unittest.TestCase):

    def setUp(self):
        self.idx = SpatialIndex()

    def _obs(self, eid, aid, x, y, z, c=0.9):
        return SpatialObservation(entity_id=eid, agent_id=aid, x=x, y=y, z=z,
                                  confidence=c, timestamp=time.time())

    def test_insert_and_count(self):
        self.idx.insert(self._obs("r1", "a", 5, 10, 0))
        self.idx.insert(self._obs("r2", "b", 20, 30, 5))
        self.assertEqual(len(self.idx.all_observations()), 2)

    def test_query_sphere(self):
        self.idx.insert(self._obs("r1", "a", 5, 5, 0))
        self.idx.insert(self._obs("r2", "b", 50, 50, 0))
        results = self.idx.query_sphere(5, 5, 0, 10.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_id, "r1")

    def test_query_sphere_empty(self):
        self.idx.insert(self._obs("r", "a", 100, 100, 0))
        results = self.idx.query_sphere(0, 0, 0, 1.0)
        self.assertEqual(len(results), 0)

    def test_query_aabb(self):
        self.idx.insert(self._obs("r1", "a", 1, 1, 0))
        self.idx.insert(self._obs("r2", "b", 10, 10, 0))
        self.idx.insert(self._obs("r3", "c", 100, 100, 0))
        results = self.idx.query_aabb(0, 0, 0, 20, 20, 20)
        self.assertEqual(len(results), 2)

    def test_query_entity(self):
        self.idx.insert(self._obs("r1", "a", 5, 5, 0))
        self.idx.insert(self._obs("r1", "b", 6, 6, 0))
        self.idx.insert(self._obs("r2", "c", 10, 10, 0))
        results = self.idx.query_entity("r1")
        self.assertEqual(len(results), 2)

    def test_query_agent(self):
        self.idx.insert(self._obs("r1", "agent-a", 5, 5, 0))
        self.idx.insert(self._obs("r2", "agent-a", 10, 10, 0))
        self.idx.insert(self._obs("r3", "agent-b", 15, 15, 0))
        results = self.idx.query_agent("agent-a")
        self.assertEqual(len(results), 2)

    def test_agent_positions(self):
        self.idx.insert(self._obs("agent-a", "agent-a", 5, 5, 0))
        self.idx.insert(self._obs("agent-a", "agent-a", 6, 6, 0))
        pos = self.idx.agent_positions()
        self.assertIn("agent-a", pos)
        self.assertEqual(pos["agent-a"].x, 6.0)

    def test_insert_dict(self):
        self.idx.insert_dict({"entity_id": "r1", "agent_id": "a",
                               "x": 5, "y": 10, "z": 0,
                               "confidence": 0.9, "timestamp": time.time()})
        self.assertEqual(len(self.idx.all_observations()), 1)

    def test_serialize_deserialize(self):
        self.idx.insert(self._obs("r1", "a", 5, 5, 0))
        self.idx.insert(self._obs("r2", "b", 10, 10, 0))
        data = self.idx.serialize()
        idx2 = SpatialIndex()
        idx2.deserialize(data)
        self.assertEqual(len(idx2.all_observations()), 2)

    def test_prune_expired(self):
        now = time.time()
        old = SpatialObservation(entity_id="old", agent_id="a",
                                 x=0, y=0, z=0, confidence=0.5,
                                 timestamp=now - 400)
        fresh = SpatialObservation(entity_id="fresh", agent_id="b",
                                   x=1, y=1, z=1, confidence=0.5,
                                   timestamp=now - 10)
        self.idx.insert(old)
        self.idx.insert(fresh)
        removed = self.idx.prune_expired(max_age_s=300.0, now=now)
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.idx.all_observations()), 1)

    def test_stats(self):
        self.idx.insert(self._obs("r1", "a", 0, 0, 0))
        self.idx.query_sphere(0, 0, 0, 10)
        stats = self.idx.stats()
        self.assertEqual(stats["inserts"], 1)

    def test_duplicate_insert_skipped(self):
        """Inserting the identical observation twice must not double-count."""
        obs = self._obs("r1", "a", 5, 5, 0)
        self.idx.insert(obs)
        self.idx.insert(obs)   # exact same object -> same dedup key
        self.assertEqual(len(self.idx.all_observations()), 1)
        stats = self.idx.stats()
        self.assertEqual(stats["inserts"], 1)
        self.assertEqual(stats["duplicates_skipped"], 1)

    def test_different_observations_not_deduped(self):
        """Distinct observations from different agents must all be kept."""
        self.idx.insert(self._obs("r1", "a", 5, 5, 0))
        self.idx.insert(self._obs("r1", "b", 5.2, 5.2, 0))
        self.assertEqual(len(self.idx.all_observations()), 2)
        self.assertEqual(self.idx.stats()["duplicates_skipped"], 0)


class TestFusion(unittest.TestCase):

    def setUp(self):
        self.idx = SpatialIndex(decay_s=300.0)

    def _obs(self, eid, aid, x, y, z, c=0.9):
        return SpatialObservation(entity_id=eid, agent_id=aid, x=x, y=y, z=z,
                                  confidence=c, timestamp=time.time())

    def test_fuse_single(self):
        self.idx.insert(self._obs("r1", "a", 5, 5, 0))
        fused = self.idx.fuse_entity("r1")
        self.assertIsNotNone(fused)
        self.assertEqual(fused.x, 5.0)

    def test_fuse_two_agents(self):
        self.idx.insert(self._obs("r1", "a", 5, 5, 0, c=0.9))
        self.idx.insert(self._obs("r1", "b", 6, 6, 0, c=0.5))
        fused = self.idx.fuse_entity("r1")
        self.assertIsNotNone(fused)
        self.assertGreater(fused.x, 5.0)
        self.assertLess(fused.x, 5.5)

    def test_fuse_all(self):
        self.idx.insert(self._obs("r1", "a", 5, 5, 0))
        self.idx.insert(self._obs("r2", "b", 10, 10, 0))
        fused = self.idx.fuse_all()
        self.assertIn("r1", fused)
        self.assertIn("r2", fused)

    def test_fuse_nonexistent(self):
        self.assertIsNone(self.idx.fuse_entity("nonexistent"))


class TestFrameTransform(unittest.TestCase):

    def test_translation(self):
        t = FrameTransform(from_frame="local", to_frame="world",
                           tx=10.0, ty=20.0, tz=0.0)
        rx, ry, rz = t.apply(15.0, 30.0, 5.0)
        self.assertEqual(rx, 5.0)
        self.assertEqual(ry, 10.0)
        self.assertEqual(rz, 5.0)


if __name__ == "__main__":
    unittest.main()