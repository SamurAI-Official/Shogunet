"""Shogunet transport tests: loopback bus semantics and link-class policy."""

import unittest

import protocol
from protocol import Envelope, ProtocolError, encode
from transports import (LINK_PROFILES, BaseTransport, LoopbackBus,
                        LoopbackTransport, profile_for)


class Collector:

    def __init__(self):
        self.frames = []

    def __call__(self, sender, frame):
        self.frames.append((sender, frame))


def env(sender, msg_type="heartbeat", recipient="*", msg_id=1):
    return Envelope(msg_id=msg_id, msg_type=msg_type, sender=sender,
                    recipient=recipient, payload={"count": 1})


class TestLoopback(unittest.TestCase):

    def setUp(self):
        self.bus = LoopbackBus("test")
        self.a = LoopbackTransport("agent-a", bus=self.bus)
        self.b = LoopbackTransport("agent-b", bus=self.bus)
        self.c = LoopbackTransport("agent-c", bus=self.bus)
        self.got_b = Collector()
        self.got_c = Collector()
        self.b.subscribe(self.got_b)
        self.c.subscribe(self.got_c)

    def test_broadcast_reaches_all_but_sender(self):
        self.assertTrue(self.a.send_frame("*", encode(env("agent-a"))))
        self.b.poll(0)
        self.c.poll(0)
        self.assertEqual(len(self.got_b.frames), 1)
        self.assertEqual(self.got_b.frames[0][0], "agent-a")
        self.assertEqual(len(self.got_c.frames), 1)

    def test_addressed_delivery(self):
        self.assertTrue(self.a.send_frame("agent-b", encode(env("agent-a"))))
        self.b.poll(0)
        self.c.poll(0)
        self.assertEqual(len(self.got_b.frames), 1)
        self.assertEqual(self.got_c.frames, [])

    def test_unknown_peer_fails(self):
        self.assertFalse(self.a.send_frame("agent-z", encode(env("agent-a"))))
        self.assertEqual(self.a.stats()["send_failed"], 1)

    def test_isolated_buses(self):
        lonely = LoopbackTransport("agent-d", bus=LoopbackBus("other"))
        got = Collector()
        lonely.subscribe(got)
        self.assertTrue(self.a.send_frame("*", encode(env("agent-a"))))
        lonely.poll(0)
        self.assertEqual(got.frames, [])

    def test_invalid_frame_refused(self):
        self.assertFalse(self.a.send_frame("*", b"garbage-not-a-frame"))
        self.assertEqual(self.a.stats()["send_failed"], 1)

    def test_bounded_queue_drops_oldest(self):
        tight = LoopbackTransport("agent-t", bus=self.bus, max_queue=2)
        got = Collector()
        tight.subscribe(got)
        for mid in (1, 2, 3):
            self.assertTrue(self.a.send_frame(
                "agent-t", encode(env("agent-a", msg_id=mid))))
        tight.poll(0)
        ids = [protocol.decode(frame).msg_id for _, frame in got.frames]
        self.assertEqual(ids, [2, 3])          # bounded: oldest dropped
        self.assertEqual(tight.stats()["dropped"], 1)

    def test_subscriber_exception_does_not_break_poll(self):
        def boom(sender, frame):
            raise RuntimeError("subscriber bug")

        self.b.subscribe(boom)
        self.assertTrue(self.a.send_frame("*", encode(env("agent-a"))))
        self.b.poll(0)                          # must not raise
        self.assertEqual(self.b.stats()["received"], 1)

    def test_stop_leaves_bus(self):
        self.b.stop()
        self.assertTrue(self.a.send_frame("*", encode(env("agent-a"))))
        self.b.poll(0)
        self.assertEqual(self.got_b.frames, [])

    def test_agent_id_required(self):
        with self.assertRaises(ValueError):
            LoopbackTransport("   ", bus=self.bus)  # stripped -> empty -> refused


class TestLinkProfiles(unittest.TestCase):

    def test_every_network_from_the_description_is_modeled(self):
        for name in ("5g", "4g", "edge", "lora", "wifi_halow", "wifi",
                     "bluetooth", "ble", "loopback"):
            self.assertIn(name, LINK_PROFILES)

    def test_constrained_profile_values(self):
        lora = profile_for("lora")
        self.assertEqual(lora.max_payload_bytes, 220)
        self.assertEqual(lora.duty_cycle, 0.01)
        self.assertFalse(lora.supports_ip)
        self.assertFalse(lora.segmented)

    def test_unknown_profile(self):
        with self.assertRaises(ProtocolError):
            profile_for("smoke_signals")


class _StubTransport(BaseTransport):
    """Minimal concrete transport for policy tests."""

    def __init__(self, profile_name):
        self.profile = profile_for(profile_name)
        self.name = "stub-%s" % profile_name

    def is_available(self):
        return True

    def start(self):
        pass

    def stop(self):
        pass

    def send_frame(self, peer, frame):
        return True

    def subscribe(self, callback):
        pass

    def poll(self, timeout=0.1):
        pass

    def stats(self):
        return {}


class TestClassEligibility(unittest.TestCase):
    """Constrained datagram media carry P0/P1 only; IP/segmented carry all."""

    def test_lora_only_low_classes(self):
        stub = _StubTransport("lora")
        self.assertTrue(stub.accepts_class(protocol.CLASS_CONTROL))
        self.assertTrue(stub.accepts_class(protocol.CLASS_MEMORY))
        self.assertFalse(stub.accepts_class(protocol.CLASS_TASK))
        self.assertFalse(stub.accepts_class(protocol.CLASS_BULK))

    def test_ble_segmentation_carries_all(self):
        stub = _StubTransport("ble")
        for cls in range(4):
            self.assertTrue(stub.accepts_class(cls))

    def test_broadband_carries_all(self):
        for name in ("wifi", "5g", "4g", "edge", "wifi_halow"):
            stub = _StubTransport(name)
            for cls in range(4):
                self.assertTrue(stub.accepts_class(cls))


if __name__ == "__main__":
    unittest.main()
