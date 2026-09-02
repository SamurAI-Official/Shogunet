"""Shogunet link simulator tests: latency, bandwidth, MTU, duty cycle, loss."""

import time
import unittest

import protocol
from link_simulator import SimulatedEther, SimulatedTransport, make_bus
from protocol import Envelope, encode
from transports import LINK_PROFILES, LinkProfile


def env(sender, msg_id, pad=0):
    return Envelope(msg_id=msg_id, msg_type="heartbeat", sender=sender,
                    payload={"count": 1, "content": "x" * pad})


def slow_profile(kbps, latency_s):
    return LinkProfile("slow", kbps=kbps, latency_s=latency_s,
                       max_payload_bytes=1400, duty_cycle=None,
                       supports_ip=True)


class TestProfiles(unittest.TestCase):

    def test_every_network_builds(self):
        for name in LINK_PROFILES:
            ether = SimulatedEther(name, seed=1)
            self.assertEqual(ether.profile.name, name)

    def test_lora_is_the_constrained_extreme(self):
        ether = SimulatedEther("lora", seed=1)
        self.assertEqual(ether.profile.max_payload_bytes, 220)
        self.assertEqual(ether.profile.duty_cycle, 0.01)
        self.assertEqual(ether.profile.kbps, 5.0)


class TestLatencyBandwidth(unittest.TestCase):

    def test_frame_arrives_only_after_latency_and_airtime(self):
        ether = SimulatedEther(slow_profile(100, 0.2), seed=7)
        tx = SimulatedTransport("agent-a", ether)
        rx = SimulatedTransport("agent-b", ether)
        got = []
        rx.subscribe(lambda s, f: got.append(f))
        ether.start()
        try:
            self.assertTrue(tx.send_frame(
                "agent-b", encode(env("agent-a", 1, pad=50))))
            rx.poll(0)
            self.assertEqual(got, [])        # latency + serialization pending
            self.assertTrue(ether.flush(3.0))
            rx.poll(0)
            self.assertEqual(len(got), 1)
            self.assertEqual(protocol.decode(got[0]).payload["count"], 1)
        finally:
            ether.close()

    def test_bandwidth_drives_delay(self):
        ether = SimulatedEther(slow_profile(10, 0.0), seed=7)
        tx = SimulatedTransport("agent-a", ether)
        rx = SimulatedTransport("agent-b", ether)
        got = []
        rx.subscribe(lambda s, f: got.append(f))
        ether.start()
        try:
            # 125-byte frame at 10 kbit/s = 100 ms of airtime
            self.assertTrue(tx.send_frame(
                "agent-b", encode(env("agent-a", 1, pad=100))))
            rx.poll(0)
            self.assertEqual(got, [])
            self.assertTrue(ether.flush(3.0))
            rx.poll(0)
            self.assertEqual(len(got), 1)
        finally:
            ether.close()


class TestImpairments(unittest.TestCase):

    def test_mtu_rejection_on_lora(self):
        ether, (tx, rx) = make_bus(["agent-a", "agent-b"], "lora", seed=1)
        try:
            self.assertFalse(tx.send_frame(
                "agent-b", encode(env("agent-a", 1, pad=400))))
            self.assertEqual(ether.stats()["dropped_mtu"], 1)
        finally:
            ether.close()

    def test_small_frame_fits_lora(self):
        ether, (tx, rx) = make_bus(["agent-a", "agent-b"], "lora", seed=1)
        got = []
        rx.subscribe(lambda s, f: got.append(f))
        try:
            self.assertTrue(tx.send_frame(
                "agent-b", encode(env("agent-a", 1, pad=50))))
            self.assertTrue(ether.flush(10.0))   # 2 s latency + airtime
            rx.poll(0)
            self.assertEqual(len(got), 1)
        finally:
            ether.close()

    def test_duty_cycle_exhaustion_on_lora(self):
        ether, (tx, rx) = make_bus(["agent-a", "agent-b"], "lora", seed=1)
        try:
            sent = 0
            for i in range(300):
                if tx.send_frame("agent-b", encode(env("agent-a", i, pad=100))):
                    sent += 1
            stats = ether.stats()
            self.assertLess(sent, 200)           # budget is ~1% airtime
            self.assertGreater(stats["dropped_duty"], 0)
        finally:
            ether.close()

    def test_loss_is_seeded_and_deterministic(self):
        counts = []
        for _ in range(2):
            ether, (tx, rx) = make_bus(["agent-a", "agent-b"], "wifi",
                                       seed=123, loss_rate=0.5)
            got = []
            rx.subscribe(lambda s, f: got.append(f))
            for i in range(100):
                tx.send_frame("*", encode(env("agent-a", i)))
            self.assertTrue(ether.flush(5.0))
            rx.poll(0)
            counts.append(len(got))
            ether.close()
        self.assertEqual(counts[0], counts[1])   # same seed, same outcome
        self.assertTrue(20 <= counts[0] <= 80)

    def test_addressed_delivery(self):
        ether, (a, b, c) = make_bus(["agent-a", "agent-b", "agent-c"],
                                    "loopback", seed=1)
        got_b, got_c = [], []
        b.subscribe(lambda s, f: got_b.append(f))
        c.subscribe(lambda s, f: got_c.append(f))
        a.send_frame("agent-b", encode(env("agent-a", 1)))
        self.assertTrue(ether.flush(1.0))
        b.poll(0)
        c.poll(0)
        self.assertEqual(len(got_b), 1)
        self.assertEqual(got_c, [])
        ether.close()

    def test_unknown_member_counted_as_dropped(self):
        ether, (tx, rx) = make_bus(["agent-a", "agent-b"], "loopback", seed=1)
        tx.send_frame("ghost", encode(env("agent-a", 1)))
        self.assertTrue(ether.flush(1.0))
        self.assertEqual(ether.stats()["dropped_no_member"], 1)
        ether.close()


if __name__ == "__main__":
    unittest.main()
