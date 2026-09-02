"""Shogunet fallback chain tests: ranking, breakers, codecs, QoS, dedup."""

import os
import tempfile
import unittest

import protocol
from protocol import (CODEC_COMPACT, CODEC_JSON, Envelope, encode,
                      segment_frame)
from store_forward import OutboxStore
from transports import BaseTransport, profile_for
from transport_fallback import TransportChain


class StubTransport(BaseTransport):
    """Scriptable transport: controllable failure, injectable inbound."""

    def __init__(self, name, profile_name, fail=False):
        self.name = name
        self.profile = profile_for(profile_name)
        self.fail = fail
        self.sent = []
        self._callbacks = []

    def is_available(self):
        return True

    def start(self):
        pass

    def stop(self):
        pass

    def send_frame(self, peer, frame):
        if self.fail:
            return False
        self.sent.append((peer, bytes(frame)))
        return True

    def subscribe(self, callback):
        self._callbacks.append(callback)

    def poll(self, timeout=0.1):
        pass

    def stats(self):
        return {}

    def inject(self, sender, frame):
        for callback in list(self._callbacks):
            callback(sender, bytes(frame))


def env(msg_id, msg_type="heartbeat", sender="agent-a", recipient="agent-b",
        pad=0):
    return Envelope(msg_id=msg_id, msg_type=msg_type, sender=sender,
                    recipient=recipient,
                    payload={"count": 1, "content": "x" * pad})


class TestFallbackChain(unittest.TestCase):

    def test_falls_back_to_healthy_transport(self):
        wifi = StubTransport("stub-wifi", "wifi", fail=True)
        lora = StubTransport("stub-lora", "lora")
        chain = TransportChain("agent-a", [wifi, lora])
        report = chain.send(env(1))
        self.assertTrue(report.ok)
        self.assertEqual(report.via, "stub-lora")
        self.assertEqual(report.codec, CODEC_COMPACT)   # constrained -> compact
        self.assertEqual(chain.stats()["sent"], 1)

    def test_breaker_opens_after_threshold(self):
        wifi = StubTransport("stub-wifi", "wifi", fail=True)
        lora = StubTransport("stub-lora", "lora")
        chain = TransportChain("agent-a", [wifi, lora],
                               breaker_threshold=3, breaker_cooldown_s=5.0)
        # Keep the failing transport preferred: it stays the attempted path
        # (its failure penalty 0.01 -> 0.51 -> 1.01 stays below lora's 5.0),
        # so consecutive failures accumulate to the breaker threshold.
        chain._circuits["stub-wifi"].latency_ewma = 0.01
        chain._circuits["stub-lora"].latency_ewma = 5.0
        for msg_id in (1, 2, 3):
            self.assertTrue(chain.send(env(msg_id)).ok)
        health = {entry["transport"]: entry for entry in chain.health()}
        self.assertTrue(health["stub-wifi"]["breaker_open"])
        self.assertEqual(health["stub-wifi"]["consecutive_failures"], 3)
        report = chain.send(env(4))
        self.assertTrue(report.ok)
        self.assertEqual(report.attempts, 1)            # only lora attempted
        self.assertEqual(report.via, "stub-lora")

    def test_constrained_media_reject_bulk_class(self):
        lora = StubTransport("stub-lora", "lora")
        chain = TransportChain("agent-a", [lora])
        report = chain.send(env(1, msg_type="bulk_snapshot"))
        self.assertFalse(report.ok)
        self.assertEqual(report.attempts, 0)
        self.assertEqual(chain.stats()["send_failed"], 1)

    def test_ble_segments_oversize_frames(self):
        ble = StubTransport("stub-ble", "ble")
        chain = TransportChain("agent-a", [ble])
        report = chain.send(env(1, msg_type="bulk_snapshot", pad=600))
        self.assertTrue(report.ok)
        self.assertTrue(report.segmented)
        self.assertGreater(len(ble.sent), 1)
        for _peer, frame in ble.sent:
            self.assertLessEqual(len(frame), ble.chunk_size())

    def test_ewma_ranking_prefers_fast_transport(self):
        wifi = StubTransport("stub-wifi", "wifi")
        lora = StubTransport("stub-lora", "lora")
        chain = TransportChain("agent-a", [wifi, lora])
        chain._circuits["stub-wifi"].latency_ewma = 5.0   # stale/slow
        chain._circuits["stub-lora"].latency_ewma = 0.1
        report = chain.send(env(1))
        self.assertEqual(report.via, "stub-lora")

    def test_json_codec_on_broadband(self):
        wifi = StubTransport("stub-wifi", "wifi")
        chain = TransportChain("agent-a", [wifi])
        report = chain.send(env(1))
        self.assertTrue(report.ok)
        self.assertEqual(report.codec, CODEC_JSON)
        self.assertEqual(protocol.decode(wifi.sent[0][1]).msg_id, 1)

class TestChainReceive(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.outbox = OutboxStore(
            os.path.join(self.tmpdir.name, "outbox.wal.jsonl"))
        self.wifi = StubTransport("stub-wifi", "wifi")
        self.chain = TransportChain("agent-a", [self.wifi],
                                    outbox=self.outbox)
        self.received = []
        self.chain.subscribe(lambda env, via: self.received.append((env, via)))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_at_least_once_persists_then_acks(self):
        report = self.chain.send(env(42, sender="agent-a",
                                     recipient="agent-b"),
                                 qos="at_least_once")
        self.assertTrue(report.ok)
        self.assertEqual(len(self.outbox), 1)           # pending until ack
        ack = Envelope(msg_id=99, msg_type="ack", sender="agent-b",
                       payload={"msg_id": 42})
        self.wifi.inject("agent-b", encode(ack))
        self.chain.poll(0)
        self.assertEqual(len(self.outbox), 0)           # ack completes it
        self.assertEqual(self.chain.stats()["acked"], 1)

    def test_inbound_delivery_and_dedup(self):
        frame = encode(env(7, sender="agent-b", recipient="agent-a"))
        self.wifi.inject("agent-b", frame)
        self.wifi.inject("agent-b", frame)              # at-least-once dupe
        self.chain.poll(0)
        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0][0].msg_id, 7)
        self.assertEqual(self.chain.stats()["duplicates"], 1)

    def test_foreign_recipient_never_reaches_handlers(self):
        # Cross-talk guard: mail addressed to another agent that lands on this
        # chain's transport (TCP/relay/broadcast) must not reach handlers.
        frame = encode(env(70, sender="agent-b", recipient="agent-c"))
        self.wifi.inject("agent-b", frame)
        self.chain.poll(0)
        self.assertEqual(self.received, [])
        self.assertEqual(self.chain.stats()["decoded"], 1)  # counted, not acted

    def test_bad_frame_counted_not_raised(self):
        self.wifi.inject("agent-b", b"garbage")
        self.chain.poll(0)
        self.assertEqual(self.chain.stats()["bad_frames"], 1)
        self.assertEqual(self.received, [])

    def test_handler_exception_does_not_break_chain(self):
        def boom(env, via):
            raise RuntimeError("handler bug")

        self.chain.subscribe(boom)
        self.wifi.inject("agent-b", encode(env(8, sender="agent-b",
                                               recipient="agent-a")))
        self.chain.poll(0)                              # must not raise
        self.assertEqual(len(self.received), 1)


class TestBroadcastCapability(unittest.TestCase):
    """Addressed-only transports are skipped for broadcasts, not punished."""

    class AddressedOnly(StubTransport):
        broadcast_supported = False

        def send_frame(self, peer, frame):
            if peer in (None, "", "*"):
                return False
            return StubTransport.send_frame(self, peer, frame)

    def test_broadcast_skips_addressed_only_without_breaker_penalty(self):
        relay_like = self.AddressedOnly("stub-relay", "4g")
        wifi = StubTransport("stub-wifi", "wifi")
        chain = TransportChain("agent-a", [relay_like, wifi])
        for _ in range(10):                      # far past breaker threshold
            report = chain.send(env(1, sender="agent-a", recipient="*"))
            self.assertTrue(report.ok)
            self.assertEqual(report.via, "stub-wifi")
        # The addressed-only transport was never attempted for a broadcast
        # and never punished: breaker closed, zero failures, zero attempts.
        health = {h["transport"]: h for h in chain.health()}
        self.assertFalse(health["stub-relay"]["breaker_open"])
        self.assertEqual(health["stub-relay"]["consecutive_failures"], 0)
        self.assertEqual(relay_like.sent, [])
        self.assertEqual(len(wifi.sent), 10)

    def test_addressed_still_uses_relay_first(self):
        relay_like = self.AddressedOnly("stub-relay", "4g")
        wifi = StubTransport("stub-wifi", "wifi")
        chain = TransportChain("agent-a", [relay_like, wifi])
        report = chain.send(env(5, sender="agent-a", recipient="agent-b"))
        self.assertTrue(report.ok)
        self.assertEqual(report.via, "stub-relay")


if __name__ == "__main__":
    unittest.main()

