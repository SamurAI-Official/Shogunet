"""Shogunet store-and-forward tests: WAL persistence, dedup, compaction."""

import os
import tempfile
import unittest

from store_forward import InboxDedup, OutboxStore


class TestOutbox(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "outbox.wal.jsonl")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_enqueue_and_pending(self):
        outbox = OutboxStore(self.path)
        self.assertTrue(outbox.enqueue(1, "agent-b", b"frame-one"))
        self.assertTrue(outbox.enqueue(2, "agent-c", b"frame-two"))
        self.assertEqual(len(outbox), 2)
        pending = outbox.pending()
        self.assertEqual([entry["msg_id"] for entry in pending], [1, 2])
        self.assertEqual(pending[0]["frame"], b"frame-one")
        self.assertEqual(pending[1]["peer"], "agent-c")

    def test_mark_done_removes(self):
        outbox = OutboxStore(self.path)
        outbox.enqueue(1, "agent-b", b"frame-one")
        self.assertTrue(outbox.mark_done(1))
        self.assertFalse(outbox.mark_done(1))     # already done
        self.assertEqual(len(outbox), 0)
        self.assertEqual(outbox.pending(), [])

    def test_wal_survives_restart(self):
        outbox = OutboxStore(self.path)
        outbox.enqueue(1, "agent-b", b"frame-one")
        outbox.enqueue(2, "agent-c", b"frame-two")
        outbox.mark_done(1)
        # Crash + restart: only undone entries replay
        revived = OutboxStore(self.path)
        self.assertEqual(len(revived), 1)
        self.assertEqual(revived.pending()[0]["msg_id"], 2)
        self.assertEqual(revived.pending()[0]["frame"], b"frame-two")

    def test_capacity_bound(self):
        outbox = OutboxStore(self.path, max_pending=3)
        for msg_id in range(5):
            self.assertEqual(outbox.enqueue(msg_id, "*", b"f"),
                             msg_id < 3)
        self.assertEqual(len(outbox), 3)

    def test_compaction_keeps_pending_only(self):
        outbox = OutboxStore(self.path)
        outbox.enqueue(1, "*", b"done-frame")
        outbox.enqueue(2, "*", b"pending-frame")
        outbox.mark_done(1)
        kept = outbox.compact()
        self.assertEqual(kept, 1)
        revived = OutboxStore(self.path)
        self.assertEqual(revived.pending()[0]["frame"], b"pending-frame")

    def test_corrupt_lines_skipped(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not json at all\n")
            handle.write("\n")
        outbox = OutboxStore(self.path)
        self.assertEqual(len(outbox), 0)


class TestInboxDedup(unittest.TestCase):

    def test_first_delivery_passes(self):
        dedup = InboxDedup()
        self.assertFalse(dedup.is_duplicate("agent-a", 1))
        self.assertFalse(dedup.is_duplicate("agent-a", 2))

    def test_duplicate_detected(self):
        dedup = InboxDedup()
        self.assertFalse(dedup.is_duplicate("agent-a", 1))
        self.assertTrue(dedup.is_duplicate("agent-a", 1))

    def test_same_id_different_sender_passes(self):
        dedup = InboxDedup()
        self.assertFalse(dedup.is_duplicate("agent-a", 1))
        self.assertFalse(dedup.is_duplicate("agent-b", 1))

    def test_capacity_evicts_oldest(self):
        dedup = InboxDedup(capacity=2)
        dedup.is_duplicate("a", 1)
        dedup.is_duplicate("a", 2)
        dedup.is_duplicate("a", 3)          # evicts (a, 1)
        self.assertFalse(dedup.is_duplicate("a", 1))
        self.assertTrue(dedup.is_duplicate("a", 3))


if __name__ == "__main__":
    unittest.main()
