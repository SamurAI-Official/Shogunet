"""Shogunet concurrency & cross-talk integration suite (Phase 6b).

Runs multiple ShugoCore-style agent instances *at the same time*: each agent
lives on its own ``TransportChain`` over a shared bus, sends from its own
thread, and we assert the hard invariants a threaded fleet must keep:

- **No cross-talk**: an addressed message reaches exactly its recipient; no
  agent ever sees mail addressed to someone else (transport + chain guards).
- **No loss**: every addressed and broadcast message is delivered exactly once.
- **Convergence**: the codependent memory mesh converges to the same fact set
  on every agent under concurrent publish, and same-key LWW storms converge.
- **Atomicity**: concurrent ``reinforce`` never loses an increment; concurrent
  WAL ``compact()`` never drops a durable record.
- **Relay isolation**: a threaded hub never leaks frames between mailboxes.

Every test is deterministic: senders start on a ``threading.Barrier`` (true
simultaneity), threads are joined, then the fleet is drained until idle before
asserting. No sleeps race the assertions.
"""

import os
import tempfile
import threading
import time
import unittest

from protocol import CODEC_COMPACT, CODEC_JSON, Envelope, encode, new_msg_id
from transports import LoopbackBus, LoopbackTransport
from transport_fallback import TransportChain
from memory_sync import InMemoryFactStore, MemorySyncNode
from store_forward import OutboxStore
from relay_server import RelayHub


def _drain_until_idle(chains, transports, sweeps=3, round_sleep=0.002,
                      max_rounds=5000):
    """Poll every chain until the fleet's delivered-count stops changing."""
    prev_total = -1
    stable = 0
    for _ in range(max_rounds):
        for chain in chains:
            chain.poll(0.0)
        total = sum(t.stats().get("received", 0) for t in transports)
        if total == prev_total:
            stable += 1
            if stable >= sweeps:
                return True
        else:
            stable = 0
        prev_total = total
        time.sleep(round_sleep)
    return False


class FakeAgent:
    """Records every envelope a chain hands it (thread-safe)."""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.events = []
        self.lock = threading.Lock()

    def on_recv(self, env, via):
        with self.lock:
            self.events.append((env, via))

    def snapshot(self):
        with self.lock:
            return list(self.events)


def _fleet(agent_ids, max_queue=8192):
    """One loopback + chain + FakeAgent per id on a shared bus."""
    bus = LoopbackBus("fleet")
    loops, chains, agents = [], [], []
    for aid in agent_ids:
        t = LoopbackTransport(aid, bus=bus, max_queue=max_queue)
        c = TransportChain(aid, [t])
        a = FakeAgent(aid)
        c.subscribe(a.on_recv)
        loops.append(t)
        chains.append(c)
        agents.append(a)
    return loops, chains, agents


class TestSimultaneousTraffic(unittest.TestCase):
    def test_threaded_fleet_no_cross_talk_no_loss_no_dup(self):
        n = 4
        addressed_per_pair = 25
        broadcasts = 10
        ids = [f"agent-{i}" for i in range(n)]
        loops, chains, agents = _fleet(ids)
        barrier = threading.Barrier(n)
        errors = []

        def sender(i):
            try:
                barrier.wait()
                peer = ids[(i + 1) % n]
                for k in range(addressed_per_pair):
                    env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                                   sender=ids[i], recipient=peer,
                                   payload={"seq": k, "kind": "addressed"})
                    if not loops[i].send_frame(peer, encode(env, CODEC_JSON)):
                        errors.append((i, "addressed-send-failed", k))
                for k in range(broadcasts):
                    env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                                   sender=ids[i], recipient="*",
                                   payload={"seq": k, "kind": "broadcast"})
                    loops[i].send_frame("*", encode(env, CODEC_JSON))
            except Exception as exc:  # barrier broken / any thread error
                errors.append((i, repr(exc)))

        threads = [threading.Thread(target=sender, args=(i,))
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        self.assertEqual(errors, [], f"sender errors: {errors}")
        self.assertTrue(_drain_until_idle(chains, loops),
                        "fleet did not quiesce")

        for i, agent in enumerate(agents):
            events = agent.snapshot()
            # 1) no cross-talk: every envelope addressed to us or the fleet
            for env, _via in events:
                self.assertIn(env.recipient, ("*", ids[i]),
                              f"{ids[i]} received cross-talk "
                              f"{env.sender}->{env.recipient}")
            # 2) exactly the right addressed batch from our dedicated peer
            peer = ids[(i - 1) % n]
            from_peer = [e for e in events
                         if e[0].sender == peer
                         and e[0].payload.get("kind") == "addressed"]
            self.assertEqual(len(from_peer), addressed_per_pair,
                             f"{ids[i]} got {len(from_peer)} addressed "
                             f"from {peer}")
            # 3) no duplicates (sender, msg_id) across all mail
            keys = [(env.sender, env.msg_id) for env, _ in events]
            self.assertEqual(len(keys), len(set(keys)),
                             f"duplicate delivery on {ids[i]}")
            # 4) every other agent's broadcasts arrived
            bcasts = [e for e in events if e[0].recipient == "*"]
            self.assertEqual(len(bcasts), (n - 1) * broadcasts,
                             f"{ids[i]} saw {len(bcasts)} broadcasts")


class TestChainCrossTalkGuard(unittest.TestCase):
    def test_foreign_recipient_dropped_broadcast_and_self_delivered(self):
        # A chain only processes mail addressed to its agent or the fleet.
        # This simulates a TCP/relay/broadcast leak where a frame for another
        # agent lands on this transport anyway.
        t = LoopbackTransport("agent-b")
        chain = TransportChain("agent-b", [t])
        seen = []
        chain.subscribe(lambda env, via: seen.append(env))

        chain._on_frame("agent-a", encode(Envelope(
            msg_id=1, msg_type="memory_fact", sender="agent-a",
            recipient="agent-c")))                    # foreign -> dropped
        self.assertEqual(seen, [])

        chain._on_frame("agent-a", encode(Envelope(
            msg_id=2, msg_type="memory_fact", sender="agent-a",
            recipient="*")))                          # fleet -> processed
        self.assertEqual(len(seen), 1)

        chain._on_frame("agent-a", encode(Envelope(
            msg_id=3, msg_type="memory_fact", sender="agent-a",
            recipient="agent-b")))                    # self -> processed
        self.assertEqual(len(seen), 2)


class TestRelayConcurrentIsolation(unittest.TestCase):
    def test_concurrent_delivery_never_leaks_between_mailboxes(self):
        n = 3
        per = 20
        hub = RelayHub(max_messages=1024)
        ids = [f"agent-{i}" for i in range(n)]
        for aid in ids:
            hub.register(aid)
        barrier = threading.Barrier(n)
        errors = []

        def deliver(i):
            try:
                barrier.wait()
                recip = ids[(i + 1) % n]
                for k in range(per):
                    env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                                   sender=ids[i], recipient=recip)
                    if not hub.deliver(recip, encode(env, CODEC_JSON))["ok"]:
                        errors.append((i, "deliver-failed", k))
            except Exception as exc:
                errors.append((i, repr(exc)))

        threads = [threading.Thread(target=deliver, args=(i,))
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        self.assertEqual(errors, [], f"deliver errors: {errors}")

        for i, aid in enumerate(ids):
            mail = hub.poll(aid, timeout_ms=0)
            msgs = mail["messages"]
            expect_sender = ids[(i - 1) % n]
            self.assertEqual(len(msgs), per,
                             f"{aid} got {len(msgs)} messages")
            for m in msgs:
                self.assertEqual(m["sender"], expect_sender,
                                 f"cross-talk into {aid} from {m['sender']}")


def _memory_fleet(agent_ids, max_queue=8192):
    """Loopback + chain + store + sync node per id on a shared bus."""
    bus = LoopbackBus("mem-fleet")
    loops, chains, stores, nodes = [], [], [], []
    for aid in agent_ids:
        t = LoopbackTransport(aid, bus=bus, max_queue=max_queue)
        c = TransportChain(aid, [t])
        s = InMemoryFactStore(aid)
        n = MemorySyncNode(aid, c, s)
        loops.append(t)
        chains.append(c)
        stores.append(s)
        nodes.append(n)
    return loops, chains, stores, nodes


class TestConcurrentMemoryConvergence(unittest.TestCase):
    def test_fleet_converges_under_concurrent_publish(self):
        n = 4
        facts_each = 5
        ids = [f"m-{i}" for i in range(n)]
        loops, chains, stores, nodes = _memory_fleet(ids)
        barrier = threading.Barrier(n)
        errors = []

        def producer(i):
            try:
                barrier.wait()
                for k in range(facts_each):
                    fact = stores[i].store_fact(
                        f"fact from {ids[i]} number {k}",
                        salience=0.5 + k * 0.1)
                    nodes[i].publish_fact(fact)
            except Exception as exc:
                errors.append((i, repr(exc)))

        threads = [threading.Thread(target=producer, args=(i,))
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        self.assertEqual(errors, [], f"producer errors: {errors}")
        self.assertTrue(_drain_until_idle(chains, loops),
                        "memory fleet did not quiesce")

        for i, store in enumerate(stores):
            self.assertEqual(store.count(), n * facts_each,
                             f"store {ids[i]} holds {store.count()} facts")
            for origin in ids:
                owned = [k for k in store._facts if k.startswith(origin + ":")]
                self.assertEqual(len(owned), facts_each,
                                 f"{ids[i]} missing facts from {origin}")


class TestConcurrentReinforcement(unittest.TestCase):
    def test_reinforce_is_atomic_under_contention(self):
        store = InMemoryFactStore("origin-a")
        key = store.store_fact("shared memory", salience=1.0,
                               fact_id=1)["key"]
        # 4x10x0.1 = 4.0 boost stays under the reinforce cap (10.0), so a
        # lost increment would show up as a shortfall in salience and count.
        threads_n, rounds = 4, 10
        barrier = threading.Barrier(threads_n)

        def worker():
            barrier.wait()
            for _ in range(rounds):
                store.reinforce(key, boost=0.1)

        threads = [threading.Thread(target=worker)
                   for _ in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        final = store.get_fact(key)
        self.assertAlmostEqual(final["salience"],
                               1.0 + threads_n * rounds * 0.1, places=6)
        self.assertEqual(final["access_count"], threads_n * rounds)


class TestConcurrentWALCompact(unittest.TestCase):
    def test_no_record_lost_during_compact_race(self):
        path = os.path.join(tempfile.mkdtemp(), "outbox.jsonl")
        sf = OutboxStore(path, max_pending=100000)
        total = 300
        stop = threading.Event()
        errors = []

        def writer():
            try:
                for mid in range(total):
                    env = Envelope(msg_id=mid, msg_type="heartbeat",
                                   sender="a", recipient="b")
                    sf.enqueue(mid, "b", encode(env, CODEC_COMPACT))
                    if mid % 5 == 0:
                        sf.compact()
            except Exception as exc:
                errors.append(repr(exc))
            finally:
                stop.set()

        t = threading.Thread(target=writer)
        t.start()
        # second thread churns compact() while the writer enqueues
        while not stop.is_set():
            sf.compact()
            time.sleep(0.0001)
        t.join(timeout=15)
        self.assertEqual(errors, [])

        replayed = OutboxStore(path, max_pending=100000)
        got = {e["msg_id"] for e in replayed.pending()}
        self.assertEqual(got, set(range(total)),
                         f"compact dropped records: missing "
                         f"{sorted(set(range(total)) - got)[:10]}...")


class TestConcurrentSameKeyConvergence(unittest.TestCase):
    def test_lww_storm_converges_across_fleet(self):
        ids = ["s-0", "s-1", "s-2"]
        loops, chains, stores, nodes = _memory_fleet(ids)
        barrier = threading.Barrier(3)

        def producer(i):
            barrier.wait()
            for v in range(10):
                fact = stores[i].store_fact(f"content v{v} from {ids[i]}",
                                            salience=float(v), fact_id=7)
                nodes[i].publish_fact(fact)

        threads = [threading.Thread(target=producer, args=(i,))
                   for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        self.assertTrue(_drain_until_idle(chains, loops),
                        "lww fleet did not quiesce")

        # each origin's fact_id 7 converges to ONE identical value fleet-wide
        for origin in ids:
            contents = []
            for store in stores:
                fact = store.get_fact(store.make_key(origin, 7))
                contents.append(fact["content"] if fact else None)
            self.assertIsNotNone(contents[0])
            self.assertEqual(contents[0], contents[1],
                             f"stores diverged on {origin}:7")
            self.assertEqual(contents[1], contents[2],
                             f"stores diverged on {origin}:7")


if __name__ == "__main__":
    unittest.main()

