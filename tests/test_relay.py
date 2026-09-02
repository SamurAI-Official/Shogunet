"""Shogunet relay tests: hub logic, HTTP API, spoof rejection, transport E2E."""

import base64
import threading
import time
import unittest

import protocol
from protocol import Envelope, encode
from relay_server import RelayHub, run_relay
from relay_transport import RelayTransport


def env(sender, msg_id, recipient="*", msg_type="heartbeat", pad=0):
    return Envelope(msg_id=msg_id, msg_type=msg_type, sender=sender,
                    recipient=recipient,
                    payload={"count": 1, "content": "x" * pad})


class TestRelayHub(unittest.TestCase):

    def setUp(self):
        self.hub = RelayHub()

    def test_register_and_list(self):
        self.assertTrue(self.hub.register("agent-a", realm="phys")["ok"])
        self.assertTrue(self.hub.is_registered("agent-a"))
        self.assertIn("agent-a", self.hub.agents())

    def test_register_requires_id(self):
        self.assertFalse(self.hub.register("   ")["ok"])

    def test_deliver_to_registered_recipient(self):
        self.hub.register("agent-a")
        self.hub.register("agent-b")
        result = self.hub.deliver("agent-b", encode(env("agent-a", 1)))
        self.assertTrue(result["ok"])
        mail = self.hub.poll("agent-b", timeout_ms=0)
        self.assertEqual(len(mail["messages"]), 1)
        self.assertEqual(mail["messages"][0]["sender"], "agent-a")
        # frame round-trips intact
        frame = base64.b64decode(mail["messages"][0]["frame_b64"])
        self.assertEqual(protocol.decode(frame).msg_id, 1)

    def test_unknown_recipient_refused(self):
        self.hub.register("agent-a")
        result = self.hub.deliver("ghost", encode(env("agent-a", 1)))
        self.assertFalse(result["ok"])
        self.assertEqual(self.hub.stats()["dropped_unknown_recipient"], 1)

    def test_spoofed_sender_refused(self):
        self.hub.register("agent-a")
        self.hub.register("agent-b")
        # frame claims sender "evil" which is not registered
        result = self.hub.deliver("agent-b", encode(env("evil", 1)))
        self.assertFalse(result["ok"])
        self.assertEqual(self.hub.stats()["dropped_spoofed"], 1)
        # a registered sender's frame passes
        self.assertTrue(self.hub.deliver("agent-b",
                                         encode(env("agent-a", 3)))["ok"])

    def test_bad_frame_refused(self):
        self.hub.register("agent-a")
        self.hub.register("agent-b")
        result = self.hub.deliver("agent-b", b"not-a-frame")
        self.assertFalse(result["ok"])
        self.assertEqual(self.hub.stats()["dropped_bad"], 1)

    def test_mailbox_full_drops(self):
        hub = RelayHub(max_messages=2)
        hub.register("agent-a")
        hub.register("agent-b")
        for msg_id in (1, 2):
            self.assertTrue(hub.deliver("agent-b",
                                        encode(env("agent-a", msg_id)))["ok"])
        self.assertFalse(hub.deliver("agent-b",
                                     encode(env("agent-a", 3)))["ok"])
        self.assertEqual(hub.stats()["dropped_full"], 1)

    def test_long_poll_wakes_on_delivery(self):
        self.hub.register("agent-a")
        self.hub.register("agent-b")
        result_box = {}

        def poller():
            result_box["result"] = self.hub.poll("agent-b", timeout_ms=3000)

        thread = threading.Thread(target=poller)
        thread.start()
        time.sleep(0.05)                    # poller is waiting
        self.assertTrue(self.hub.deliver("agent-b",
                                         encode(env("agent-a", 9)))["ok"])
        thread.join(timeout=3.0)
        self.assertIn("result", result_box)
        self.assertEqual(result_box["result"]["messages"][0]["msg_id"], 9)

    def test_message_ttl_expiry(self):
        hub = RelayHub(message_ttl_s=0.15)
        hub.register("agent-a")
        hub.register("agent-b")
        hub.deliver("agent-b", encode(env("agent-a", 1)))
        time.sleep(0.3)
        mail = hub.poll("agent-b", timeout_ms=0)
        self.assertEqual(mail["messages"], [])
        self.assertEqual(hub.stats()["expired"], 1)

class _LiveRelayTestCase(unittest.TestCase):
    """Base class: a live HTTP relay on an ephemeral port."""

    def setUp(self):
        self.hub = RelayHub()
        self.server, self.thread = run_relay(self.hub, "127.0.0.1", 0)
        port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()


class TestRelayHTTP(_LiveRelayTestCase):
    """HTTP-level checks via the same hub the transport talks to."""

    def test_register_deliver_poll_over_http(self):
        self.hub.register("agent-a")
        self.hub.register("agent-b")
        self.assertTrue(self.hub.deliver(
            "agent-b", encode(env("agent-a", 5)))["ok"])
        mail = self.hub.poll("agent-b", timeout_ms=0)
        self.assertEqual(len(mail["messages"]), 1)


class TestRelayTransport(_LiveRelayTestCase):

    def _wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_end_to_end_delivery(self):
        a = RelayTransport("agent-a", self.base_url,
                           poll_timeout_ms=200, poll_retry_s=0.1)
        b = RelayTransport("agent-b", self.base_url,
                           poll_timeout_ms=200, poll_retry_s=0.1)
        got_b = []
        b.subscribe(lambda sender, frame: got_b.append((sender, frame)))
        a.start()
        b.start()
        try:
            self.assertTrue(self._wait_until(
                lambda: a.is_available() and b.is_available()))
            frame = encode(env("agent-a", 42, recipient="agent-b"))
            self.assertTrue(a.send_frame("agent-b", frame))
            self.assertTrue(self._wait_until(
                lambda: len(b._inbox) >= 1 or len(got_b) >= 1))
            b.poll(0)
            self.assertEqual(len(got_b), 1)
            sender, got = got_b[0]
            self.assertEqual(sender, "agent-a")
            self.assertEqual(protocol.decode(got).msg_id, 42)
        finally:
            a.stop()
            b.stop()

    def test_broadcast_not_supported_on_relay(self):
        a = RelayTransport("agent-a", self.base_url,
                           poll_timeout_ms=200, poll_retry_s=0.1)
        a.start()
        try:
            self.assertTrue(self._wait_until(a.is_available))
            self.assertFalse(a.send_frame("*", encode(env("agent-a", 1))))
            self.assertEqual(a.stats()["send_failed"], 1)
        finally:
            a.stop()

    def test_send_to_unregistered_recipient_fails(self):
        a = RelayTransport("agent-a", self.base_url,
                           poll_timeout_ms=200, poll_retry_s=0.1)
        a.start()
        try:
            self.assertTrue(self._wait_until(a.is_available))
            self.assertFalse(a.send_frame("ghost",
                                          encode(env("agent-a", 1))))
        finally:
            a.stop()


if __name__ == "__main__":
    unittest.main()