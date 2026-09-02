"""Phase 6 hardening: multi-agent integration, partition storms, chaos."""

import json
import os
import tempfile
import time
import unittest

from protocol import (
    CODEC_COMPACT, CODEC_JSON, Envelope, decode, encode, new_msg_id,
)
from transports import LoopbackTransport, LoopbackBus
from lora_transport import LoraTransport, create_pipe_pair as create_lora_pair
from bluetooth_transport import BluetoothTransport, create_bluetooth_pipe_pair
from transport_fallback import TransportChain
from agent_registry import AgentRegistry
from store_forward import OutboxStore
from memory_sync import MemorySyncNode, InMemoryFactStore
from mesh_query import MeshQuery
from audit import AuditChain


def _drain(transports, rounds=40):
    for _ in range(rounds):
        for t in transports:
            if hasattr(t, "poll"):
                t.poll(0.01)
        time.sleep(0.005)


def _decode_sub(loop, agent):
    """Subscribe ``agent`` to ``loop`` decoding wire frames into envelopes."""
    def on_frame(sender, frame):
        env = decode(frame)
        agent.on_recv(env, sender or env.sender)
    loop.subscribe(on_frame)
    return on_frame


class FakeAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.received = []
        self.memory = {}
        self.audit = AuditChain(
            os.path.join(tempfile.mkdtemp(), "audit.jsonl"))

    def on_recv(self, env, sender):
        self.received.append((env, sender))
        if env.msg_type == "memory_fact":
            self.memory[env.payload.get("fact_id")] = env.payload


def _fleet(agent_ids):
    """Shared-bus loopback fleet; returns (agents, loopbacks, chains, stores, nodes)."""
    bus = LoopbackBus("fleet")
    agents = [FakeAgent(aid) for aid in agent_ids]
    loops = [LoopbackTransport(a.agent_id, bus=bus) for a in agents]
    chains = [TransportChain(a.agent_id, [loops[i]]) for i, a in enumerate(agents)]
    stores = [InMemoryFactStore(a.agent_id) for a in agents]
    nodes = [MemorySyncNode(a.agent_id, chains[i], stores[i])
             for i, a in enumerate(agents)]
    return agents, loops, chains, stores, nodes


class TestMultiAgentFleet(unittest.TestCase):
    def test_three_agents_loopback(self):
        agents = [FakeAgent(f"agent-{i}") for i in range(3)]
        bus = LoopbackBus("t3")
        loops = [LoopbackTransport(a.agent_id, bus=bus) for a in agents]
        for i, me in enumerate(loops):
            _decode_sub(me, agents[i])
        env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                       sender="agent-0", recipient="*")
        loops[0].send_frame("*", encode(env, CODEC_JSON))
        _drain(loops)
        self.assertEqual(len(agents[1].received), 1)
        self.assertEqual(len(agents[2].received), 1)
        self.assertEqual(agents[1].received[0][0].msg_type, "heartbeat")

    def test_five_agents_mixed_transports(self):
        agents = [FakeAgent(f"node-{i}") for i in range(5)]
        bus = LoopbackBus("t5")
        l0 = LoopbackTransport("node-0", bus=bus)
        l1 = LoopbackTransport("node-1", bus=bus)
        _decode_sub(l0, agents[0])
        _decode_sub(l1, agents[1])
        lora_a, lora_b = create_lora_pair()
        lt2 = LoraTransport("node-2", port=lora_a, chunk_size=120)
        lt3 = LoraTransport("node-3", port=lora_b, chunk_size=120)
        lt2.on_recv(agents[2].on_recv)
        lt3.on_recv(agents[3].on_recv)
        ble_p, _ = create_bluetooth_pipe_pair()
        bt4 = BluetoothTransport("node-4", port=ble_p, mode="ble", ble_mtu=100)
        bt4.on_recv(agents[4].on_recv)
        all_t = [l0, l1, lt2, lt3, bt4]
        fact_env = Envelope(msg_id=new_msg_id(), msg_type="memory_fact",
                            sender="node-0", recipient="*",
                            payload={"fact_id": "shared-fact", "content": "hello fleet",
                                     "salience": 0.7, "origin": "node-0"})
        l0.send_frame("*", encode(fact_env, CODEC_JSON))
        _drain(all_t)
        self.assertIn("shared-fact", agents[1].memory)
        self.assertEqual(agents[1].memory["shared-fact"]["content"], "hello fleet")

    def test_memory_sync_across_fleet(self):
        agents, loops, _chains, stores, nodes = _fleet(["m-0", "m-1", "m-2", "m-3"])
        fact = stores[0].store_fact("collaborative memory", salience=0.8)
        nodes[0].publish_fact(fact)
        _drain(loops)
        for i in range(1, 4):
            got = [f["fact_id"] for f in stores[i]._facts.values()]
            self.assertIn(fact["fact_id"], got)
            stored = next(f for f in stores[i]._facts.values()
                          if f["fact_id"] == fact["fact_id"])
            self.assertEqual(stored["content"], "collaborative memory")


class TestPartitionStorms(unittest.TestCase):
    def test_partition_heals_with_replay(self):
        path = os.path.join(tempfile.mkdtemp(), "outbox.jsonl")
        sf = OutboxStore(path, max_pending=100)
        for i in range(5):
            env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                           sender="peer", recipient="partition-agent",
                           payload={"seq": i})
            sf.enqueue(env.msg_id, env.sender, encode(env, CODEC_COMPACT))
        self.assertEqual(len(sf), 5)
        pending = sf.pending()
        self.assertEqual(len(pending), 5)
        for entry in pending:
            self.assertTrue(sf.mark_done(entry["msg_id"]))
        self.assertEqual(len(sf), 0)

    def test_partition_replay_survives_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "outbox.jsonl")
        sf = OutboxStore(path, max_pending=100)
        env = Envelope(msg_id=42, msg_type="heartbeat", sender="peer",
                       recipient="partition-agent", payload={"seq": 0})
        self.assertTrue(sf.enqueue(env.msg_id, env.sender, encode(env, CODEC_COMPACT)))
        # Simulate crash: fresh store object on the same WAL path.
        sf2 = OutboxStore(path, max_pending=100)
        self.assertEqual(len(sf2), 1)
        replayed = [entry for entry in sf2.pending()]
        decoded = decode(replayed[0]["frame"])
        self.assertEqual(decoded.msg_id, 42)
        self.assertEqual(decoded.msg_type, "heartbeat")

    def test_conflict_storm_detected(self):
        store = InMemoryFactStore("conflict-agent")
        store.store_fact("v1", salience=0.5, fact_id=1)
        for i in range(20):
            existing = store.get_fact(store.make_key("conflict-agent", 1))
            if existing["content"] != f"v{i}":
                store.store_fact(f"v{i}", salience=0.1 * i, fact_id=1)
        fact = store.get_fact(store.make_key("conflict-agent", 1))
        self.assertEqual(fact["content"], "v19")
        self.assertEqual(store.count(), 1)


class TestSyncReplayDeterminism(unittest.TestCase):
    def test_sync_produces_same_state(self):
        s1 = InMemoryFactStore("det-1")
        s2 = InMemoryFactStore("det-2")
        ops = [
            ("store", "f1", "content-a", 0.5),
            ("store", "f2", "content-b", 0.6),
            ("reinforce", "f1", 0.1),
            ("store", "f3", "content-c", 0.9),
            ("reinforce", "f2", 0.2),
        ]
        for op in ops:
            if op[0] == "store":
                s1.store_fact(op[2], salience=op[3], fact_id=int(op[1][1:]))
                s2.store_fact(op[2], salience=op[3], fact_id=int(op[1][1:]))
            elif op[0] == "reinforce":
                key = s1.make_key("det-1", int(op[1][1:]))
                s1.reinforce(key, boost=op[2])
                key2 = s2.make_key("det-2", int(op[1][1:]))
                s2.reinforce(key2, boost=op[2])
        # Same fact ids, same salience ordering (digests embed origin, so
        # compare per-fact salience keys instead of digest lists).
        d1 = sorted(s1.digests(), key=lambda d: d["fact_id"])
        d2 = sorted(s2.digests(), key=lambda d: d["fact_id"])
        self.assertEqual([d["fact_id"] for d in d1], [d["fact_id"] for d in d2])
        self.assertEqual([d["s"] for d in d1], [d["s"] for d in d2])

    def test_audit_chain_deterministic(self):
        a1 = AuditChain(os.path.join(tempfile.mkdtemp(), "a1.jsonl"))
        a2 = AuditChain(os.path.join(tempfile.mkdtemp(), "a2.jsonl"))
        events = [
            ("send", {"msg_id": 1, "type": "heartbeat"}),
            ("recv", {"msg_id": 2, "type": "reinforce"}),
            ("send", {"msg_id": 3, "type": "memory_fact"}),
        ]
        for action, data in events:
            a1.append(action, data)
            a2.append(action, data)
        self.assertEqual(a1.tail, a2.tail)

    def test_audit_chain_tamper_detected(self):
        path = os.path.join(tempfile.mkdtemp(), "a.jsonl")
        a = AuditChain(path)
        a.append("send", {"msg_id": 1})
        a.append("recv", {"msg_id": 2})
        self.assertEqual(a.verify(), [])
        # Tamper the on-disk chain: rewrite the first record's payload.
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        record = json.loads(lines[0])
        record["payload"]["msg_id"] = 999
        lines[0] = json.dumps(record, sort_keys=True) + "\n"
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        self.assertNotEqual(a.verify(), [])


class TestFallbackChainIntegration(unittest.TestCase):
    def test_priority_routing_lora_vs_loopback(self):
        a = FakeAgent("route-agent")
        bus = LoopbackBus("route-bus")
        peer_loop = LoopbackTransport("peer", bus=bus)
        peer_got = FakeAgent("peer")
        _decode_sub(peer_loop, peer_got)
        loop = LoopbackTransport("route-agent", bus=bus)
        lora_a, lora_b = create_lora_pair()
        lora = LoraTransport("route-agent", port=lora_a, chunk_size=120)
        chain = TransportChain("route-agent", [lora, loop], audit=a.audit)
        chain.subscribe(a.on_recv)
        # P0 control is eligible on the constrained link first; it must send.
        ctrl = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                        sender="peer", recipient="route-agent")
        self.assertTrue(chain.send(ctrl).ok)
        # P2 task is refused by LoRa -> the chain must fall back to loopback.
        task = Envelope(msg_id=new_msg_id(), msg_type="task_request",
                        sender="route-agent", recipient="peer",
                        payload={"task": "do something"})
        report = chain.send(task)
        self.assertTrue(report.ok)
        _drain([loop, peer_loop])
        self.assertEqual(len(peer_got.received), 1)
        self.assertEqual(peer_got.received[0][0].msg_type, "task_request")

    def test_registry_pairing_enforced(self):
        reg = AgentRegistry()
        self.assertFalse(reg.is_paired("unknown-peer"))
        reg.pair("peer-1", manifest={"realm": "phys"})
        self.assertTrue(reg.is_paired("peer-1"))
        reg._paired["peer-1"]["expires_at"] = time.time() - 1
        self.assertFalse(reg.is_paired("peer-1"))


class TestMeshQueryIntegration(unittest.TestCase):
    def test_mesh_query_fanout(self):
        from transports import LoopbackBus

        bus = LoopbackBus("iq")

        def make(agent_id):
            t = LoopbackTransport(agent_id, bus=bus)
            c = TransportChain(agent_id, [t])
            s = InMemoryFactStore(agent_id)
            m = MeshQuery(agent_id, c, s)
            return t, c, s, m

        _ta, _ca, sa, ma = make("query-agent")
        tb, _cb, sb, _mb = make("responder")

        def pump_all():
            _ca.poll(0)
            _cb.poll(0)

        sb.store_fact("the berminator is at grid 42", salience=0.9)
        results = ma.query("berminator", peers=["responder"],
                           timeout_s=1.0, poller=pump_all)
        self.assertEqual(len(results), 1)
        hit = results[0]
        self.assertEqual(hit["origin"], "responder")
        self.assertIn("grid 42", hit["content"])


if __name__ == "__main__":
    unittest.main()

