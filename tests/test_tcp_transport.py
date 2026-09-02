"""Shogunet TCP transport tests: handshake, framing, delivery, refusal."""

import socket
import struct
import time
import unittest

import protocol
from protocol import Envelope, encode
from tcp_transport import TCPTransport


def env(sender, msg_type="heartbeat", msg_id=1, recipient="*"):
    return Envelope(msg_id=msg_id, msg_type=msg_type, sender=sender,
                    recipient=recipient, payload={"count": 1})


class Collector:

    def __init__(self):
        self.frames = []

    def __call__(self, sender, frame):
        self.frames.append((sender, frame))


class TestTCPTransport(unittest.TestCase):

    def setUp(self):
        self.a = TCPTransport("agent-a", listen_host="127.0.0.1",
                              listen_port=0)
        self.b = TCPTransport("agent-b", listen_host="127.0.0.1",
                              listen_port=0)
        self.a.start()
        self.b.start()
        self.got_b = Collector()
        self.got_a = Collector()
        self.b.subscribe(self.got_b)
        self.a.subscribe(self.got_a)

    def tearDown(self):
        self.a.stop()
        self.b.stop()

    def _poll_until(self, collector, count, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.b.poll(0)
            self.a.poll(0)
            if len(collector.frames) >= count:
                return True
            time.sleep(0.02)
        return False

    def test_handshake_then_delivery(self):
        self.a.add_peer("agent-b", "127.0.0.1", self.b.local_port)
        self.assertTrue(self.a.send_frame(
            "agent-b", encode(env("agent-a", msg_id=7))))
        self.assertTrue(self._poll_until(self.got_b, 2))
        # First frame is the announce handshake, second the heartbeat.
        first = protocol.decode(self.got_b.frames[0][1])
        second = protocol.decode(self.got_b.frames[1][1])
        self.assertEqual(first.msg_type, "announce")
        self.assertEqual(first.sender, "agent-a")
        self.assertEqual(second.msg_type, "heartbeat")
        self.assertEqual(second.msg_id, 7)
        self.assertEqual(self.got_b.frames[1][0], "agent-a")

    def test_bidirectional_over_single_dial(self):
        # a dials b once; both sides then know each other's dial address
        # (from the announce handshake) and share one bidirectional stream.
        self.a.add_peer("agent-b", "127.0.0.1", self.b.local_port)
        self.assertTrue(self.a.send_frame(
            "agent-b", encode(env("agent-a", msg_id=1))))
        self.assertTrue(self._poll_until(self.got_b, 2))
        # b replies over the accepted connection -- no dial required.
        self.assertTrue(self.b.send_frame(
            "agent-a", encode(env("agent-b", msg_id=2))))
        self.assertTrue(self._poll_until(self.got_a, 1))
        reply = protocol.decode(self.got_a.frames[-1][1])
        self.assertEqual(reply.msg_type, "heartbeat")
        self.assertEqual(reply.sender, "agent-b")
        self.assertEqual(self.b.stats()["dials"], 0)
        # each side learned the other's listen address from the announce
        self.assertIn("agent-b", self.a._peers)
        self.assertIn("agent-a", self.b._peers)

    def test_invalid_frame_refused(self):
        self.a.add_peer("agent-b", "127.0.0.1", self.b.local_port)
        self.assertFalse(self.a.send_frame("agent-b", b"junk"))
        self.assertEqual(self.a.stats()["send_failed"], 1)

    def test_unknown_peer_fails(self):
        self.assertFalse(self.a.send_frame("agent-z", encode(env("agent-a"))))
        self.assertEqual(self.a.stats()["send_failed"], 1)

    def test_handshake_refused_without_announce(self):
        # A raw connection that speaks before identifying is cut off.
        sock = socket.create_connection(("127.0.0.1", self.b.local_port),
                                        timeout=2.0)
        intruder = encode(env("intruder", msg_type="heartbeat", msg_id=9))
        sock.sendall(struct.pack(">I", len(intruder)) + intruder)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.b.stats()["handshake_refused"] >= 1:
                break
            time.sleep(0.02)
        self.assertGreaterEqual(self.b.stats()["handshake_refused"], 1)
        # The server must terminate the connection: observe EOF or error.
        sock.settimeout(2.0)
        try:
            data = sock.recv(64)
        except (OSError, ConnectionError):
            data = b""      # reset / refused is equally terminal
        self.assertEqual(data, b"")
        sock.close()


if __name__ == "__main__":
    unittest.main()
