"""Shogunet codependent memory mesh tests: sync, reinforce, tombstones,
profiles, anti-entropy digests."""

import unittest

from memory_sync import (InMemoryFactStore, MemorySyncNode, SharingProfile,
                         _cosine, _embed, fact_digest)
from transport_fallback import TransportChain


def make_agent(agent_id, bus):
    """Loopback transport + chain + fact store + sync node."""
    from transports import LoopbackTransport
    transport = LoopbackTransport(agent_id, bus=bus)
    chain = TransportChain(agent_id, [transport])
    store = InMemoryFactStore(agent_id)
    node = MemorySyncNode(agent_id, chain, store)
    return transport, chain, store, node


class TwoAgentFixture(unittest.TestCase):

    def setUp(self):
        from transports import LoopbackBus
        self.bus = LoopbackBus("mem")
        self.ta, self.ca, self.sa, self.na = make_agent("agent-a", self.bus)
        self.tb, self.cb, self.sb, self.nb = make_agent("agent-b", self.bus)

    def pump(self, rounds=3):
        for _ in range(rounds):
            self.ca.poll(0)
            self.cb.poll(0)


class TestEmbedding(unittest.TestCase):

    def test_deterministic_across_instances(self):
        vec1 = _embed("the robot turned left and saw the wall")
        vec2 = _embed("the robot turned left and saw the wall")
        self.assertEqual(vec1, vec2)

    def test_similar_texts_rank_above_unrelated(self):
        a = _embed("robot navigation obstacle avoidance")
        b = _embed("robot navigation obstacle avoidance")
        c = _embed("cooking pasta carbonara recipe")
        self.assertGreater(_cosine(a, b), _cosine(a, c))

    def test_fact_digest_stable(self):
        self.assertEqual(fact_digest("agent-a", 7), fact_digest("agent-a", 7))
        self.assertNotEqual(fact_digest("agent-a", 7), fact_digest("agent-a", 8))


class TestFactStore(unittest.TestCase):

    def setUp(self):
        self.store = InMemoryFactStore("agent-a")
        self.fact = self.store.store_fact(
            "warehouse bay 3 door code is 4821", kind="procedure",
            salience=1.5)

    def test_store_and_get(self):
        self.assertEqual(self.store.count(), 1)
        got = self.store.get_fact(self.fact["key"])
        self.assertEqual(got["content"], "warehouse bay 3 door code is 4821")
        self.assertEqual(got["origin"], "agent-a")
        self.assertEqual(self.store.facts_by_kind("procedure"), [got])

    def test_search_ranks_by_similarity(self):
        self.store.store_fact("unrelated cooking recipe notes")
        hits = self.store.search("warehouse door code")
        self.assertEqual(hits[0]["key"], self.fact["key"])

    def test_reinforce_raises_salience_and_access(self):
        before = self.store.get_fact(self.fact["key"])
        self.store.reinforce(self.fact["key"], boost=0.25)
        after = self.store.get_fact(self.fact["key"])
        self.assertGreater(after["salience"], before["salience"])
        self.assertEqual(after["access_count"], 1)

    def test_foreign_fact_keyed_by_origin(self):
        foreign = self.store.store_fact("shared intel", origin="agent-b",
                                        fact_id=99)
        self.assertEqual(foreign["key"], "agent-b:99")
        self.assertIsNotNone(self.store.get_fact("agent-b:99"))

    def test_remove(self):
        self.assertTrue(self.store.remove(self.fact["key"]))
        self.assertEqual(self.store.count(), 0)

class TestSyncPropagation(TwoAgentFixture):

    def test_fact_replicates_from_a_to_b(self):
        fact = self.sa.store_fact("relay tower at grid 4-7", salience=1.2)
        self.na.publish_fact(fact)
        self.pump()
        self.assertEqual(self.sb.count(), 1)
        got = self.sb.get_fact("agent-a:1")     # keyed by origin
        self.assertEqual(got["content"], "relay tower at grid 4-7")
        self.assertEqual(got["origin"], "agent-a")
        self.assertEqual(got["salience"], 1.2)

    def test_no_relay_storm(self):
        # B never republishes a fact it learned from A.
        fact = self.sa.store_fact("scout observation zulu 22")
        self.na.publish_fact(fact)
        self.pump()
        self.nb.publish_fact(self.sb.get_fact("agent-a:1"))
        self.pump()
        self.assertEqual(self.na.stats()["facts_received"], 0)


class TestReinforcementLoop(TwoAgentFixture):

    def test_reinforce_boosts_origin_salience(self):
        fact = self.sa.store_fact("optimal crop row spacing is 0.9m",
                                  salience=1.0)
        self.na.publish_fact(fact)
        self.pump()
        before = self.sa.get_fact(fact["key"])["salience"]
        # B found A's fact useful -> reinforce flows back to A
        self.nb.send_reinforce("agent-a", "agent-a", fact["fact_id"],
                               amount=0.5)
        self.pump()
        after = self.sa.get_fact(fact["key"])["salience"]
        self.assertGreater(after, before)
        self.assertEqual(self.na.stats()["reinforces_applied"], 1)

    def test_reinforce_to_non_origin_is_ignored(self):
        # B's reinforce claiming origin agent-b does nothing at A, because
        # only the origin applies reinforcement.
        self.nb.send_reinforce("agent-b", "agent-b", 7, amount=1.0)
        self.pump()
        self.assertEqual(self.na.stats()["reinforces_applied"], 0)


class TestTombstonePruning(TwoAgentFixture):

    def test_origin_prune_propagates(self):
        fact = self.sa.store_fact("transient reading 7.2", salience=0.1)
        self.na.publish_fact(fact)
        self.pump()
        self.assertEqual(self.sb.count(), 1)
        # A prunes locally, then broadcasts the tombstone.
        self.sa.remove(fact["key"])
        self.na.publish_tombstone("agent-a", fact["fact_id"],
                                  reason="decayed")
        self.pump()
        self.assertEqual(self.sb.count(), 0)
        self.assertEqual(self.nb.stats()["tombstones_applied"], 1)

    def test_stale_peer_cannot_resurrect(self):
        fact = self.sa.store_fact("count once", salience=0.05)
        self.na.publish_fact(fact)
        self.pump()
        self.sa.remove(fact["key"])
        self.na.publish_tombstone("agent-a", fact["fact_id"])
        self.pump()
        self.assertEqual(self.sb.count(), 0)


class TestSharingProfiles(TwoAgentFixture):

    def test_private_peer_never_syncs(self):
        # Outbound policy lives with the publisher: A marks B private.
        self.na.set_profile("agent-b", SharingProfile.PRIVATE)
        fact = self.sa.store_fact("classified route plan")
        self.na.publish_fact(fact, peers=["agent-b"])
        self.pump()
        self.assertEqual(self.sb.count(), 0)
        self.assertEqual(self.na.stats()["facts_sent"], 0)

    def test_redacted_peer_still_receives(self):
        self.nb.set_profile("agent-a", SharingProfile.REDACTED)
        fact = self.sa.store_fact("payload passes through redaction")
        self.na.publish_fact(fact, peers=["agent-b"])
        self.pump()
        self.assertEqual(self.sb.count(), 1)


class TestAntiEntropy(TwoAgentFixture):

    def test_digest_pulls_missing_facts(self):
        # A and B each hold a local fact. A sends its digests to B; B replies
        # with the facts A is missing (origin B).
        fact_a = self.sa.store_fact("beacon channel 38", salience=2.0)
        self.sb.store_fact("relay password is salmon", salience=1.8)
        self.na.publish_fact(fact_a, peers=["agent-b"])
        self.pump()
        self.na.send_digests("agent-b")
        self.pump()
        # A now holds both its own fact and B's (pulled via anti-entropy).
        self.assertEqual(self.sa.count(), 2)
        self.assertIsNotNone(self.sa.get_fact("agent-b:1"))
        self.assertEqual(self.nb.stats()["facts_pulled"], 1)


if __name__ == "__main__":
    unittest.main()