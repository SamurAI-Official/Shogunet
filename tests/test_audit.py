"""Shogunet audit chain tests: chaining, tamper detection, reload, CLI."""

import os
import tempfile
import unittest

from audit import AuditChain, GENESIS_HASH


class TestAuditChain(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_append_chains_hashes(self):
        chain = AuditChain(self.path)
        first = chain.append("test_event", {"k": 1})
        second = chain.append("test_event", {"k": 2})
        self.assertEqual(first["prev"], GENESIS_HASH)
        self.assertEqual(second["prev"], first["hash"])
        self.assertEqual(len(first["hash"]), 64)
        self.assertEqual(chain.verify(), [])

    def test_verify_detects_tampered_payload(self):
        chain = AuditChain(self.path)
        chain.append("pairing", {"agent": "agent-a"})
        # Tamper with the payload but leave the hash: verification must flag it.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"seq": 99, "ts": "x", "event": "forged",'
                         ' "payload": {}, "prev": "%s", "hash": "%s"}\n'
                         % (chain.tail, "f" * 64))
        problems = chain.verify()
        self.assertTrue(any("hash mismatch" in p["reason"] for p in problems))

    def test_verify_detects_broken_link(self):
        chain = AuditChain(self.path)
        chain.append("a", {"n": 1})
        chain.append("b", {"n": 2})
        # Rewrite the file with the second record's prev corrupted.
        lines = []
        with open(self.path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        import json
        second = json.loads(lines[1])
        second["prev"] = "0" * 64
        lines[1] = json.dumps(second, sort_keys=True) + "\n"
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        problems = chain.verify()
        self.assertTrue(any("broken link" in p["reason"] for p in problems))

    def test_verify_detects_corrupt_line(self):
        chain = AuditChain(self.path)
        chain.append("a", {"n": 1})
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("{definitely not json\n")
        problems = chain.verify()
        self.assertTrue(any(p["reason"] == "corrupt json" for p in problems))

    def test_reload_adopts_tail(self):
        chain = AuditChain(self.path)
        chain.append("a", {"n": 1})
        revived = AuditChain(self.path)
        self.assertEqual(revived.tail, chain.tail)
        self.assertEqual(len(revived), 1)
        self.assertEqual(revived.verify(), [])

    def test_append_never_raises_on_io_failure(self):
        chain = AuditChain("/nonexistent-dir/audit.jsonl")
        record = chain.append("x", {"n": 1})   # must not raise
        self.assertEqual(record["event"], "x")


if __name__ == "__main__":
    unittest.main()