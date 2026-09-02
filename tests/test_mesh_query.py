"""Shogunet mesh query tests: fan-out, provenance labels, merge, profiles."""

import unittest

from memory_sync import InMemoryFactStore, SharingProfile
from mesh_query import MeshQuery
from transport_fallback import TransportChain


def make_agent(agent_id, bus):
    from transports import LoopbackTransport
    transport = LoopbackTransport(agent_id, bus=bus)
    chain = TransportChain(agent_id, [transport])
    store = InMemoryFactStore(agent_id)
    mesh = MeshQuery(agent_id, chain, store)
    return transport, chain, store, mesh


class MeshFixture(unittest.TestCase):

    def setUp(self):
        from transports import LoopbackBus
        self.bus = LoopbackBus("mesh")
        self.ta, self.ca, self.sa, self.ma = make_agent("agent-a", self.bus)
        self.tb, self.cb, self.sb, self.mb = make_agent("agent-b", self.bus)
        self.tc, self.cc, self.sc, self.mc = make_agent("agent-c", self.bus)

    def pump(self):
        self.ca.poll(0)
        self.cb.poll(0)
        self.cc.poll(0)

    def pump_all(self):
        """Fleet-wide receive pump used as the query() poller."""
        self.pump()


class TestMeshQuery(MeshFixture):

    def test_single_peer_returns_provenance_labeled_results(self):
        self.sb.store_fact("warehouse door pin is 4821", salience=1.5)
        results = self.ma.query("warehouse door pin",
                                peers=["agent-b"], timeout_s=1.0, poller=self.pump_all)
        self.pump()
        self.assertEqual(len(results), 1)
        hit = results[0]
        self.assertEqual(hit["origin"], "agent-b")
        self.assertEqual(hit["peer"], "agent-b")
        self.assertEqual(hit["fact_id"], 1)
        self.assertIn("4821", hit["content"])
        self.assertEqual(self.mb.stats()["queries_answered"], 1)

    def test_fan_out_merges_across_peers(self):
        self.sb.store_fact("north gate frequency 446.1", salience=1.2)
        self.sc.store_fact("south gate frequency 448.3", salience=1.1)
        results = self.ma.query("gate frequency",
                                peers=["agent-b", "agent-c"],
                                timeout_s=1.0, poller=self.pump_all)
        self.pump()
        origins = {r["origin"] for r in results}
        self.assertEqual(origins, {"agent-b", "agent-c"})
        self.assertTrue(all("frequency" in r["content"] for r in results))

    def test_dedup_same_fact_from_shared_sources(self):
        # Both B and C hold the same fact (same origin + fact_id).
        for store in (self.sb, self.sc):
            store.store_fact("shared source memo", origin="agent-b",
                             fact_id=7)
        results = self.ma.query("source memo",
                                peers=["agent-b", "agent-c"],
                                timeout_s=1.0, poller=self.pump_all)
        self.pump()
        keys = [(r["origin"], r["fact_id"]) for r in results]
        self.assertEqual(len(keys), len(set(keys)))

    def test_private_peer_never_queried(self):
        self.mb.set_profile("agent-a", SharingProfile.PRIVATE)
        self.sb.store_fact("must stay hidden", salience=3.0)
        results = self.ma.query("hidden", peers=["agent-b"], timeout_s=0.3)
        self.pump()
        self.assertEqual(results, [])
        self.assertEqual(self.mb.stats()["queries_answered"], 0)

    def test_no_peers_returns_empty(self):
        self.assertEqual(self.ma.query("anything", peers=[]), [])

    def test_results_scored_and_capped(self):
        self.sb.store_fact("emergency channel 911", salience=3.0)
        self.sb.store_fact("document scanner", salience=0.2)
        results = self.ma.query("emergency", peers=["agent-b"],
                                top_k=1, timeout_s=1.0, poller=self.pump_all)
        self.pump()
        self.assertEqual(len(results), 1)
        self.assertIn("911", results[0]["content"])


class TestMeshQueryOverConstrainedLink(MeshFixture):

    def test_query_rejected_on_lora_class(self):
        # memory_query is P2 (task class) -- a LoRa-only chain cannot carry it,
        # so the queue refuses and we get no results (no crash).
        from link_simulator import SimulatedEther, SimulatedTransport
        ether = SimulatedEther("lora", seed=1)
        ta = SimulatedTransport("agent-a", ether)
        tb = SimulatedTransport("agent-b", ether)
        ca = TransportChain("agent-a", [ta])
        cb = TransportChain("agent-b", [tb])
        sa = InMemoryFactStore("agent-a")
        sb = InMemoryFactStore("agent-b")
        ma = MeshQuery("agent-a", ca, sa)
        mb = MeshQuery("agent-b", cb, sb)
        sb.store_fact("lora cant carry queries", salience=2.0)
        try:
            results = ma.query("lora", peers=["agent-b"], timeout_s=0.5)
            cb.poll(0)
            self.assertEqual(results, [])
            self.assertEqual(ma.stats()["queries_sent"], 1)
        finally:
            ether.close()


if __name__ == "__main__":
    unittest.main()