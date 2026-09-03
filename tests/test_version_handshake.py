"""Shogunet 0.4.0 version handshake tests: compat rule, TCP admission
refusal for incompatible clients, roster version surfacing."""

import json
import os
import time
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import version
from host import ShugonetHost
from agent_runtime import ShugonetAgentRuntime
from protocol import Envelope, encode, new_msg_id
from tcp_transport import TCPTransport
from transport_fallback import TransportChain


def audit_events(audit):
    try:
        with open(audit.path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except OSError:
        return []


def make_client(aid, host):
    client = TCPTransport(aid, listen=False, profile="wifi")
    client.add_peer(host.agent_id, "127.0.0.1", host.tcp_port)
    client._default_peer = host.agent_id
    client.start()
    return client


def dial_once(client, aid):
    chain = TransportChain(aid, [client])
    chain.subscribe(lambda env, via: None)
    env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                   sender=aid, recipient="*", topic="/shugunet/x/y",
                   payload={})
    chain.send(env, qos="best_effort")
    return chain


class TestCompatRule(unittest.TestCase):

    def test_same_minor_is_compatible(self):
        self.assertTrue(version.is_compatible(version.VERSION))
        self.assertTrue(version.is_compatible("0.4.9"))

    def test_older_minor_refused(self):
        self.assertFalse(version.is_compatible("0.3.9"))
        self.assertFalse(version.is_compatible("0.1.0"))

    def test_newer_minor_refused(self):
        # The host is the compatibility floor: a 0.5 client speaks a wire
        # dialect the 0.4 host has never seen.
        self.assertFalse(version.is_compatible("0.5.0"))

    def test_garbage_refused(self):
        self.assertFalse(version.is_compatible(""))
        self.assertFalse(version.is_compatible("not-a-version"))
        self.assertFalse(version.is_compatible(None))


class TestHandshakeEnforcement(unittest.TestCase):

    def setUp(self):
        self.host = ShugonetHost(agent_id="ver-host", tcp_port=0, relay_port=0,
                                 heartbeat_timeout_s=2.0)
        self.host.start()

    def tearDown(self):
        self.host.stop()

    def test_incompatible_client_refused_and_audited(self):
        aid = "agent-old"
        self.host.pair(aid)
        # Build a client transport whose announce claims an old version.
        client = make_client(aid, self.host)
        try:
            stale = Envelope(msg_id=new_msg_id(), msg_type="announce",
                             sender=aid, recipient="*",
                             payload={"port": 0,
                                      "shugonet_version": "0.3.9"})
            client._announce_frame = encode(stale)
            dial_once(client, aid)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self.host.tcp.stats().get("handshake_refused", 0) >= 1:
                    break
                time.sleep(0.05)
            self.assertGreaterEqual(
                self.host.tcp.stats()["handshake_refused"], 1)
            mismatches = [r for r in audit_events(self.host.audit)
                          if r["event"] == "version_mismatch"]
            self.assertTrue(mismatches)
            self.assertEqual(mismatches[-1]["payload"]["client_version"],
                             "0.3.9")
        finally:
            client.stop()

    def test_compatible_client_admitted_with_version_on_roster(self):
        aid = "agent-cur"
        self.host.pair(aid)
        runtime = ShugonetAgentRuntime(
            agent_id=aid, host_tcp_host="127.0.0.1",
            host_tcp_port=self.host.tcp_port,
            host_agent_id=self.host.agent_id,
            host_relay_url=self.host.relay_url, realm="sim")
        runtime.connect_to_host()
        try:
            deadline = time.time() + 5.0
            status = {}
            while time.time() < deadline:
                status = self.host.status()
                if status["alive_count"] >= 1:
                    break
                time.sleep(0.05)
            self.assertEqual(status["alive_count"], 1)
            self.assertEqual(status["roster"][0]["client_version"],
                             version.VERSION)
            self.assertEqual(status["shugonet_version"], version.VERSION)
            self.assertEqual(status["protocol_version"],
                             version.PROTOCOL_VERSION)
        finally:
            runtime.stop()

    def test_client_without_version_reported_unknown(self):
        # A peer that sends no version (pre-0.4.0 or hand-rolled) is admitted
        # but flagged "unknown" on the roster.
        aid = "agent-blank"
        self.host.pair(aid)
        client = make_client(aid, self.host)
        try:
            blank = Envelope(msg_id=new_msg_id(), msg_type="announce",
                             sender=aid, recipient="*", payload={"port": 0})
            client._announce_frame = encode(blank)
            dial_once(client, aid)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self.host.tcp.stats().get("connections", 0) >= 1:
                    break
                time.sleep(0.05)
            self.assertEqual(self.host.tcp.peer_versions().get(aid), "unknown")
        finally:
            client.stop()


if __name__ == "__main__":
    unittest.main()
