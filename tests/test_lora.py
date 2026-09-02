"""Tests for LoRa transport (Phase 2C -- constrained long-range links)."""

import time
import unittest

from protocol import (
    CLASS_CONTROL, CLASS_MEMORY, CLASS_TASK, CODEC_COMPACT, Envelope,
    SegmentAssembler, decode, encode, new_msg_id, segment_frame,
)
from lora_transport import LoraTransport, create_pipe_pair


class TestSerialPipe(unittest.TestCase):
    def test_round_trip(self):
        a, b = create_pipe_pair()
        a.write(b"hello")
        self.assertEqual(b.read(), b"hello")

    def test_read_partial(self):
        a, b = create_pipe_pair()
        a.write(b"hello world")
        self.assertEqual(b.read(5), b"hello")
        self.assertEqual(b.read(), b" world")

    def test_close_stops_write(self):
        a, b = create_pipe_pair()
        a.close()
        self.assertEqual(b.write(b"x"), 0)


class TestLoraTransport(unittest.TestCase):
    def _make_pair(self, chunk_size=120):
        a_port, b_port = create_pipe_pair()
        a = LoraTransport("agent-a", port=a_port, chunk_size=chunk_size)
        b = LoraTransport("agent-b", port=b_port, chunk_size=chunk_size)
        return a, b

    def _control_frame(self):
        env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                       sender="agent-a", recipient="agent-b")
        return encode(env, codec=CODEC_COMPACT)

    def _memory_frame(self):
        env = Envelope(msg_id=new_msg_id(), msg_type="reinforce",
                       sender="agent-a", recipient="agent-b",
                       payload={"fact_id": "f1", "amount": 0.5})
        return encode(env, codec=CODEC_COMPACT)

    def _task_frame(self):
        env = Envelope(msg_id=new_msg_id(), msg_type="task_request",
                       sender="agent-a", recipient="agent-b",
                       payload={"task": "do something"})
        return encode(env, codec=CODEC_COMPACT)

    def test_control_frame_round_trip(self):
        a, b = self._make_pair()
        received = []
        b.on_recv(lambda env, sender: received.append(env))
        frame = self._control_frame()
        self.assertTrue(a.send_frame(frame))
        for _ in range(20):
            a.poll()
            b.poll()
            if received:
                break
            time.sleep(0.01)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].msg_type, "heartbeat")
        self.assertEqual(received[0].sender, "agent-a")

    def test_memory_frame_round_trip(self):
        a, b = self._make_pair()
        received = []
        b.on_recv(lambda env, sender: received.append((env, sender)))
        frame = self._memory_frame()
        self.assertTrue(a.send_frame(frame))
        for _ in range(20):
            a.poll()
            b.poll()
            if received:
                break
            time.sleep(0.01)
        self.assertEqual(len(received), 1)
        env, sender = received[0]
        self.assertEqual(env.msg_type, "reinforce")
        self.assertEqual(env.payload["fact_id"], "f1")
        self.assertAlmostEqual(env.payload["amount"], 0.5)

    def test_task_frame_refused(self):
        a, b = self._make_pair()
        frame = self._task_frame()
        self.assertFalse(a.send_frame(frame))

    def test_segmentation_large_frame(self):
        a, b = self._make_pair(chunk_size=60)
        received = []
        b.on_recv(lambda env, sender: received.append(env))
        env = Envelope(msg_id=new_msg_id(), msg_type="memory_fact",
                       sender="agent-a", recipient="agent-b",
                       payload={"fact_id": "big-fact", "content": "x" * 500})
        frame = encode(env, codec=CODEC_COMPACT)
        self.assertTrue(len(frame) > 60)
        self.assertTrue(a.send_frame(frame))
        for _ in range(50):
            a.poll()
            b.poll()
            if received:
                break
            time.sleep(0.01)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].msg_type, "memory_fact")
        self.assertEqual(received[0].payload["fact_id"], "big-fact")
        self.assertEqual(len(received[0].payload["content"]), 500)

    def test_send_envelope_convenience(self):
        a, b = self._make_pair()
        received = []
        b.on_recv(lambda env, sender: received.append(env))
        env = Envelope(msg_id=new_msg_id(), msg_type="tombstone",
                       sender="agent-a", recipient="agent-b",
                       payload={"fact_id": "gone"})
        self.assertTrue(a.send_envelope(env))
        for _ in range(20):
            a.poll()
            b.poll()
            if received:
                break
            time.sleep(0.01)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].msg_type, "tombstone")

    def test_peers_tracked(self):
        a, b = self._make_pair()
        received = []
        b.on_recv(lambda env, sender: received.append(env))
        frame = self._control_frame()
        a.send_frame(frame)

    def test_audit_events(self):
        events = []

        class FakeAudit:
            def append(self, action, data):
                events.append((action, dict(data)))

        a, b = self._make_pair()
        a._audit = FakeAudit()
        b._audit = FakeAudit()
        received = []
        b.on_recv(lambda env, sender: received.append(env))
        frame = self._control_frame()
        a.send_frame(frame)
        for _ in range(20):
            a.poll()
            b.poll()
            if received:
                break
            time.sleep(0.01)
        actions = [e[0] for e in events]
        self.assertIn("lora_send", actions)
        self.assertIn("lora_recv", actions)

    def test_invalid_frame_rejected(self):
        a, b = self._make_pair()
        received = []
        b.on_recv(lambda env, sender: received.append(env))
        self.assertFalse(a.send_frame(b"short"))
        bad = bytearray(self._control_frame())
        bad[5] = 99
        self.assertFalse(a.send_frame(bytes(bad)))

    def test_is_available(self):
        a, _ = self._make_pair()
        self.assertTrue(a.is_available())
        empty = LoraTransport("agent-x")
        self.assertFalse(empty.is_available())

    def test_start_no_port_raises(self):
        a = LoraTransport("agent-x")
        with self.assertRaises(RuntimeError):
            a.start()

    def test_outbox_full_drops(self):
        a, b = self._make_pair()
        a._max_queue = 1
        self.assertTrue(a.send_frame(self._control_frame()))
        self.assertFalse(a.send_frame(self._control_frame()))

    def test_duty_cycle_enforcement(self):
        a, b = self._make_pair()
        a._max_airtime = 0.001
        a._duty_window = 10.0
        a._consume_airtime(100000)
        frame = self._control_frame()
        a.send_frame(frame)
        a._drain_outbox()
        self.assertEqual(len(a._outbox), 1)


class TestSegmentationProtocol(unittest.TestCase):
    def test_segment_and_reassemble(self):
        env = Envelope(msg_id=new_msg_id(), msg_type="memory_fact",
                       sender="a", recipient="b",
                       payload={"content": "x" * 1000})
        frame = encode(env, codec=CODEC_COMPACT)
        chunks = segment_frame(frame, 80)
        self.assertGreater(len(chunks), 1)
        assembler = SegmentAssembler()
        result = None
        for chunk in chunks[:-1]:
            self.assertIsNone(assembler.add(chunk))
        result = assembler.add(chunks[-1])
        self.assertIsNotNone(result)
        decoded = decode(result)
        self.assertEqual(decoded.payload["content"], "x" * 1000)

    def test_small_frame_not_segmented(self):
        env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                       sender="a", recipient="b")
        frame = encode(env, codec=CODEC_COMPACT)
        chunks = segment_frame(frame, 500)
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()

