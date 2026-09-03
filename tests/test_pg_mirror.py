"""PgFactMirror tests (fail-closed behavior). Live-DB round-trip tests run
only when a reachable PostgreSQL DSN is provided via SHUGONET_PG_DSN."""

import os
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pg_store


@unittest.skipUnless(pg_store._HAS_PSYCOPG,
                     "psycopg2 not installed (pip install 'shugonet[postgres]')")
class TestPgFactMirrorFailClosed(unittest.TestCase):

    def test_bad_table_name_refused(self):
        with self.assertRaises(ValueError):
            pg_store.PgFactMirror("dbname=x", table="1_bad; DROP TABLE x")

    def test_unreachable_dsn_raises(self):
        # Fail-closed: no silent stub mode when the database is unreachable.
        dsn = ("host=127.0.0.1 port=1 dbname=shugonet_missing "
               "connect_timeout=1")
        with self.assertRaises(Exception):
            pg_store.PgFactMirror(dsn)

    def test_import_failure_hint(self):
        self.assertIn("shugonet[postgres]", pg_store._INSTALL_HINT)


@unittest.skipUnless(os.environ.get("SHUGONET_PG_DSN"),
                     "set SHUGONET_PG_DSN to run live pg round-trips")
class TestPgFactMirrorLive(unittest.TestCase):

    def _mirror(self):
        return pg_store.PgFactMirror(os.environ["SHUGONET_PG_DSN"])

    def test_upsert_remove_roundtrip(self):
        from memory_sync import InMemoryFactStore, hashed_embedding
        mirror = self._mirror()
        try:
            store = InMemoryFactStore("agent-a")
            fact = store.store_fact("live pg roundtrip fact")
            mirror.upsert_fact(fact)
            self.assertGreaterEqual(mirror.count(), 1)
            fact2 = store.store_fact("live pg roundtrip fact",
                                     fact_id=fact["fact_id"],
                                     salience=2.0)
            mirror.upsert_fact(fact2)   # upsert: same key, salience GREATEST
            self.assertTrue(mirror.remove(fact["key"]))
            self.assertFalse(mirror.remove(fact["key"]))
        finally:
            mirror.close()

    def test_backend_hook_writes_through(self):
        from transports import LoopbackBus, LoopbackTransport
        from memory_sync import InMemoryFactStore, MemorySyncNode
        from transport_fallback import TransportChain
        mirror = self._mirror()
        try:
            bus = LoopbackBus("pg-hook")
            ta = LoopbackTransport("agent-a", bus=bus)
            tb = LoopbackTransport("agent-b", bus=bus)
            ca = TransportChain("agent-a", [ta])
            cb = TransportChain("agent-b", [tb])
            na = MemorySyncNode("agent-a", ca, InMemoryFactStore("agent-a"))
            nb = MemorySyncNode("agent-b", cb,
                                InMemoryFactStore("agent-b"), backend=mirror)
            fact = na.store.store_fact("pg write-through via mesh")
            na.publish_fact(fact)
            for _ in range(3):
                ca.poll(0)
                cb.poll(0)
            self.assertTrue(mirror.remove(fact["key"]))
        finally:
            mirror.close()


if __name__ == "__main__":
    unittest.main()
