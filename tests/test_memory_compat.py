"""Shugonet 0.4.0 <-> ShugoCore 1.4.0 memory compatibility tests.

Fact-schema parity with SemanticMemory/PgSemanticMemory columns, exact
embedding parity with vector_db.hashed_embedding, packed wire vectors,
and the MemorySyncNode persistence-backend hook.
"""

import base64
import importlib.util
import os
import time
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_sync import (InMemoryFactStore, MAX_VECTOR_DIM, MemorySyncNode,
                         _cosine, _embed, hashed_embedding, pack_vector,
                         unpack_vector)
from transport_fallback import TransportChain


def shugocore_dir():
    """Locate a ShugoCore checkout (env var first, then sibling dirs)."""
    candidates = [os.environ.get("SHUGOCORE_PATH"),
                  os.path.join(os.path.dirname(
                      os.path.dirname(os.path.abspath(__file__))),
                      "..", "Shugocore"),
                  os.path.join(os.path.dirname(
                      os.path.dirname(os.path.abspath(__file__))),
                      "..", "ShugoCore")]
    for path in candidates:
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    return None


class TestFactSchemaParity(unittest.TestCase):

    def test_fact_carries_semanticmemory_columns(self):
        store = InMemoryFactStore("agent-a")
        fact = store.store_fact("bolt driver torque specification", kind="spec",
                                metadata={"source": "manual"})
        # The columns PgSemanticMemory persists must all survive the mesh.
        for column in ("content", "kind", "salience", "access_count",
                       "metadata", "created_at", "last_accessed"):
            self.assertIn(column, fact)
        self.assertEqual(fact["kind"], "spec")
        self.assertEqual(fact["metadata"], {"source": "manual"})

    def test_search_touches_last_accessed(self):
        store = InMemoryFactStore("agent-a")
        store.store_fact("navigation waypoint tolerance")
        before = store.search("navigation")[0]["last_accessed"]
        time.sleep(0.01)
        after = store.search("navigation")[0]["last_accessed"]
        self.assertGreater(after, before)

    def test_created_at_survives_the_wire_hop(self):
        captured = []

        class CaptureChain:
            def subscribe(self, cb):
                pass

            def send(self, env, qos=None):
                captured.append(env)
                return type("R", (), {"ok": True, "via": "t"})()

        store_a = InMemoryFactStore("agent-a")
        node_a = MemorySyncNode("agent-a", CaptureChain(), store_a)
        past = time.time() - 3600.0
        fact = store_a.store_fact("legacy fact from the archive",
                                  created_at=past)
        node_a.publish_fact(fact)
        self.assertTrue(captured)

        store_b = InMemoryFactStore("agent-b")
        node_b = MemorySyncNode("agent-b", CaptureChain(), store_b)
        received = [e for e in captured if e.msg_type == "memory_fact"]
        node_b._on_fact(received[-1])
        key = store_b.make_key("agent-a", fact["fact_id"])
        synced = store_b.get_fact(key)
        self.assertIsNotNone(synced)
        self.assertAlmostEqual(synced["created_at"], past, delta=1.0)


class TestEmbeddingParity(unittest.TestCase):

    def test_dimension_default_matches_shugocore(self):
        vec = _embed("anything at all")
        self.assertEqual(len(vec), MAX_VECTOR_DIM)
        self.assertEqual(len(vec), 256)

    def test_parity_with_shugocore_hashed_embedding(self):
        core_dir = shugocore_dir()
        if not core_dir:
            self.skipTest("ShugoCore checkout not found")
        spec = importlib.util.spec_from_file_location(
            "shugocore_vector_db", os.path.join(core_dir, "vector_db.py"))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            self.skipTest("ShugoCore vector_db import failed")
        texts = [
            "The robot turned left and saw the wall",
            "cooking pasta carbonara recipe with basil",
            "TORQUE-BOLT 42 spec! (rev 7)",
            "multiple   spaces and MiXeD CaSe tokens 123",
            "",
        ]
        for text in texts:
            ours = hashed_embedding(text, 256)
            theirs = module.hashed_embedding(text, 256)
            self.assertEqual(ours, theirs, f"parity failed for {text!r}")

    def test_alias_still_works(self):
        self.assertEqual(_embed("parity check"),
                         hashed_embedding("parity check", 256))


class TestPackedWireVector(unittest.TestCase):

    def test_roundtrip(self):
        vec = hashed_embedding("wire packing roundtrip", 256)
        packed = pack_vector(vec)
        self.assertIsInstance(packed, str)
        self.assertLessEqual(len(packed), 2048)   # inside MAX_STR_VALUE
        unpacked = unpack_vector(packed)
        self.assertEqual(len(unpacked), 256)
        for a, b in zip(vec, unpacked):
            self.assertAlmostEqual(a, b, delta=1e-6)

    def test_corrupt_input_is_none(self):
        self.assertIsNone(unpack_vector(None))
        self.assertIsNone(unpack_vector(""))
        self.assertIsNone(unpack_vector("not base64!!"))
        self.assertIsNone(unpack_vector(base64.b64encode(b"odd").decode()))


class _RecorderBackend:
    """Duck-typed MemorySyncNode backend that records every call."""

    def __init__(self):
        self.upserts = []
        self.removes = []

    def upsert_fact(self, fact):
        self.upserts.append(dict(fact))

    def remove(self, key):
        self.removes.append(str(key))


class TestPersistenceBackendHook(unittest.TestCase):

    def _fixture(self, backend_b):
        from transports import LoopbackBus, LoopbackTransport
        bus = LoopbackBus("backend-hook")
        ta = LoopbackTransport("agent-a", bus=bus)
        tb = LoopbackTransport("agent-b", bus=bus)
        ca = TransportChain("agent-a", [ta])
        cb = TransportChain("agent-b", [tb])
        sa = InMemoryFactStore("agent-a")
        sb = InMemoryFactStore("agent-b")
        na = MemorySyncNode("agent-a", ca, sa)
        nb = MemorySyncNode("agent-b", cb, sb, backend=backend_b)
        return ca, cb, na, nb

    def _pump(self, ca, cb, rounds=3):
        for _ in range(rounds):
            ca.poll(0)
            cb.poll(0)

    def test_applied_fact_is_mirrored_to_backend(self):
        backend = _RecorderBackend()
        ca, cb, na, nb = self._fixture(backend)
        fact = na.store.store_fact("mirror this fact to persistence",
                                   kind="note")
        na.publish_fact(fact)
        self._pump(ca, cb)
        self.assertTrue(backend.upserts)
        mirrored = backend.upserts[-1]
        self.assertEqual(mirrored["content"], fact["content"])
        self.assertEqual(mirrored["origin"], "agent-a")
        self.assertEqual(mirrored["kind"], "note")
        self.assertIn("created_at", mirrored)

    def test_tombstone_removes_from_backend(self):
        backend = _RecorderBackend()
        ca, cb, na, nb = self._fixture(backend)
        fact = na.store.store_fact("fact that will be forgotten")
        na.publish_fact(fact)
        self._pump(ca, cb)
        na.publish_tombstone("agent-a", fact["fact_id"], reason="test prune")
        self._pump(ca, cb)
        self.assertTrue(backend.removes)
        self.assertIn(backend.removes[-1],
                      [f"agent-a:{fact['fact_id']}"])

    def test_reinforce_updates_backend_salience(self):
        # Reinforce flows back to the ORIGIN (node A), so A carries the
        # backend here: its mirror must show the boosted salience.
        backend = _RecorderBackend()
        ca, cb, na, nb = self._fixture(None)
        # Rebuild with backends: A mirrors reinforce application, B mirrors
        # mesh fact application.
        from transports import LoopbackBus, LoopbackTransport
        bus = LoopbackBus("backend-hook2")
        ta = LoopbackTransport("agent-a", bus=bus)
        tb = LoopbackTransport("agent-b", bus=bus)
        ca = TransportChain("agent-a", [ta])
        cb = TransportChain("agent-b", [tb])
        na = MemorySyncNode("agent-a", ca, InMemoryFactStore("agent-a"),
                            backend=backend)
        nb = MemorySyncNode("agent-b", cb, InMemoryFactStore("agent-b"))
        fact = na.store.store_fact("fact the peer will reinforce")
        na.publish_fact(fact)
        self._pump(ca, cb)
        nb.send_reinforce("agent-a", "agent-a", fact["fact_id"], amount=1.0)
        self._pump(ca, cb)
        self._pump(ca, cb)
        upserts = [u for u in backend.upserts
                   if u["content"] == fact["content"]]
        self.assertGreaterEqual(len(upserts), 1)
        self.assertGreater(upserts[-1]["salience"], fact["salience"])

    def test_backend_failure_never_breaks_sync(self):
        class ExplodingBackend:
            def upsert_fact(self, fact):
                raise RuntimeError("pg down")

            def remove(self, key):
                raise RuntimeError("pg down")

        from transports import LoopbackBus, LoopbackTransport
        bus = LoopbackBus("backend-fail")
        ta = LoopbackTransport("agent-a", bus=bus)
        tb = LoopbackTransport("agent-b", bus=bus)
        ca = TransportChain("agent-a", [ta])
        cb = TransportChain("agent-b", [tb])
        na = MemorySyncNode("agent-a", ca, InMemoryFactStore("agent-a"))
        nb = MemorySyncNode("agent-b", cb, InMemoryFactStore("agent-b"),
                            backend=ExplodingBackend())
        fact = na.store.store_fact("facts still land with pg down")
        na.publish_fact(fact)
        for _ in range(3):
            ca.poll(0)
            cb.poll(0)
        key = nb.store.make_key("agent-a", fact["fact_id"])
        self.assertIsNotNone(nb.store.get_fact(key))


if __name__ == "__main__":
    unittest.main()
