"""Tests for Bluetooth transport (Phase 2C -- short-range links)."""

import time
import unittest

from protocol import CODEC_COMPACT, Envelope, encode, new_msg_id
from bluetooth_transport import BluetoothTransport, create_bluetooth_pipe_pair


def _drain(a, b, received, rounds=30):
    for _ in range(rounds):
        a.poll()
        b.poll()
        if received:
            return True
        time.sleep(0.01)
    return False


class TestBytePipe(unittest.TestCase):
    def test_round_trip(self):
        a, b = create_bluetooth_pipe_pair()
        a.write(b"hello")
        self.assertEqual(b.read(), b"hello")

    def test_read_partial(self):
        a, b = create_bluetooth_pipe_pair()
        a.write(b"hello world")
        self.assertEqual(b.read(5), b"hello")

    def test_close_stops_write(self):
        a, b = create_bluetooth_pipe_pair()
        a.close()
        self.assertEqual(b.write(b"x"), 0)


class TestBluetoothTransport(unittest.TestCase):
    def _make(self, mode="ble", mtu=23):
        ap, bp = create_bluetooth_pipe_pair()
        a = BluetoothTransport("aa", port=ap, mode=mode, ble_mtu=mtu)
        b = BluetoothTransport("ab", port=bp, mode=mode, ble_mtu=mtu)
        return a, b

    def _frame(self, t, pl=None):
        return encode(Envelope(msg_id=new_msg_id(), msg_type=t, sender="aa",
                                  recipient="ab", payload=pl or {}),
                       codec=CODEC_COMPACT)

    def test_ble_control(self):
        a, b = self._make()
        r = []
        b.on_recv(lambda e, s: r.append(e))
        self.assertTrue(a.send_frame(self._frame("heartbeat")))
        _drain(a, b, r)
        self.assertEqual(r[0].msg_type, "heartbeat")
        self.assertEqual(r[0].sender, "aa")

    def test_ble_memory(self):
        a, b = self._make()
        r = []
        b.on_recv(lambda e, s: r.append(e))
        self.assertTrue(a.send_frame(self._frame("reinforce", {"fact_id": "f1", "amount": 0.5})))
        _drain(a, b, r)
        self.assertEqual(r[0].msg_type, "reinforce")
        self.assertEqual(r[0].payload["fact_id"], "f1")

    def test_ble_segmentation(self):
        a, b = self._make(mtu=100)
        r = []
        b.on_recv(lambda e, s: r.append(e))
        env = Envelope(msg_id=new_msg_id(), msg_type="memory_fact", sender="aa",
                       recipient="ab", payload={"fact_id": "bf", "content": "y" * 300})
        self.assertTrue(a.send_frame(encode(env, CODEC_COMPACT)))
        _drain(a, b, r, rounds=100)
        self.assertEqual(r[0].payload["fact_id"], "bf")
        self.assertEqual(len(r[0].payload["content"]), 300)

    def test_rfcomm_round_trip(self):
        a, b = self._make(mode="rfcomm")
        r = []
        b.on_recv(lambda e, s: r.append(e))
        self.assertTrue(a.send_frame(self._frame("heartbeat")))
        _drain(a, b, r)
        self.assertEqual(r[0].msg_type, "heartbeat")

    def test_bulk_refused(self):
        a, b = self._make()
        self.assertFalse(a.send_frame(self._frame("bulk_snapshot")))

    def test_task_accepted(self):
        a, b = self._make()
        self.assertTrue(a.send_frame(self._frame("task_request", {"task": "x"})))

    def test_send_envelope(self):
        a, b = self._make(mtu=100)
        r = []
        b.on_recv(lambda e, s: r.append(e))
        self.assertTrue(a.send_envelope(Envelope(msg_id=new_msg_id(), msg_type="tombstone",
                                                sender="aa", recipient="ab", payload={"fact_id": "g"})))
        _drain(a, b, r)
        self.assertEqual(r[0].msg_type, "tombstone")

    def test_peers(self):
        a, b = self._make(mtu=100)
        r = []
        b.on_recv(lambda e, s: r.append(e))
        a.send_frame(self._frame("heartbeat"))
        _drain(a, b, r)
        self.assertIn("aa", b.peers())

    def test_outbox_full(self):
        a, b = self._make(mtu=100)
        a._max_queue = 1
        self.assertTrue(a.send_frame(self._frame("heartbeat")))
        self.assertFalse(a.send_frame(self._frame("heartbeat")))

    def test_audit(self):
        ev = []
        class FA:
            def append(self, act, d):
                ev.append(act)
        a, b = self._make(mtu=100)
        a._audit, b._audit = FA(), FA()
        r = []
        b.on_recv(lambda e, s: r.append(e))
        a.send_frame(self._frame("heartbeat"))
        _drain(a, b, r)
        self.assertIn("bt_send", ev)
        self.assertIn("bt_recv", ev)

    def test_is_available(self):
        a, _ = self._make()
        self.assertTrue(a.is_available())
        self.assertFalse(BluetoothTransport("x").is_available())

    def test_start_no_port(self):
        with self.assertRaises(RuntimeError):
            BluetoothTransport("x").start()

    def test_invalid_mode(self):
        p, _ = create_bluetooth_pipe_pair()
        with self.assertRaises(ValueError):
            BluetoothTransport("x", port=p, mode="bad")


    def test_invalid_frame_rejected(self):
        a, b = self._make(mtu=100)
        r = []
        b.on_recv(lambda e, s: r.append(e))
        self.assertFalse(a.send_frame(b"short"))
        bad = bytearray(self._frame("heartbeat"))
        bad[5] = 99
        self.assertFalse(a.send_frame(bytes(bad)))


if __name__ == "__main__":
    unittest.main()
