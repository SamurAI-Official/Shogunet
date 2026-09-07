import logging
import os
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import version
from audit import AuditChain
from memory_sync import InMemoryFactStore, MemorySyncNode
from mesh_query import MeshQuery
from protocol import Envelope, new_msg_id
from relay_transport import RelayTransport
from spatial import SpatialIndex
from spatial_sync import SpatialMemoryNode
from store_forward import OutboxStore
from tcp_transport import TCPTransport
from transport_fallback import TransportChain

logger = logging.getLogger(__name__)


class ShugonetAgentRuntime:
    def __init__(self, agent_id, host_tcp_host="127.0.0.1", host_tcp_port=0,
                 host_agent_id=None, host_relay_url=None, realm="phys",
                 manifest=None, store=None, audit=None, outbox_path=None,
                 on_message=None):
        self.agent_id = agent_id
        self.host_tcp_host = host_tcp_host
        self.host_tcp_port = int(host_tcp_port)
        self.host_agent_id = host_agent_id or "shugonet-host"
        self.host_relay_url = host_relay_url
        self.realm = realm
        self.manifest = dict(manifest or {})
        self.manifest.setdefault("realm", realm)
        # Version handshake for the relay path: the host surfaces this from
        # the join manifest (TCP peers report via the announce frame instead).
        self.manifest.setdefault("shugonet_version", version.VERSION)
        self.store = store or InMemoryFactStore(agent_id)
        self.audit = audit
        self.on_message = on_message
        if outbox_path is None:
            outbox_path = os.path.join(tempfile.mkdtemp(), f"agent-{agent_id}-outbox.jsonl")
        self.outbox = OutboxStore(outbox_path)
        self._running = False
        self._stop = threading.Event()
        self._loop_thread = None
        self.chain = None
        self.mesh = None
        self.sync_node = None
        self.spatial = SpatialIndex()
        self.spatial_sync = None
        self._own_position = None  # (x, y, z, frame_id) tuple or None
        self._stats = {"sent": 0, "received": 0, "errors": 0}
        self._lock = threading.Lock()

    def connect_to_host(self):
        if self._running:
            return
        self.tcp = TCPTransport(self.agent_id, listen=False, profile="wifi",
                                audit=self.audit)
        self.tcp.add_peer(self.host_agent_id, self.host_tcp_host,
                          self.host_tcp_port)
        self.tcp._default_peer = self.host_agent_id
        self.tcp.start()
        transports = [self.tcp]
        self.relay = None
        if self.host_relay_url:
            self.relay = RelayTransport(
                self.agent_id, hub_url=self.host_relay_url,
                realm=self.realm, manifest=self.manifest)
            self.relay.start()
            transports.append(self.relay)
        self.chain = TransportChain(
            self.agent_id, transports,
            outbox=self.outbox, audit=self.audit)
        self.chain.subscribe(self._on_envelope)
        self.sync_node = MemorySyncNode(
            self.agent_id, self.chain, self.store)
        self.mesh = MeshQuery(self.agent_id, self.chain, self.store)
        self.spatial_sync = SpatialMemoryNode(
            self.agent_id, self.chain, self.spatial)
        self._running = True
        self._stop.clear()
        self._loop_thread = threading.Thread(
            target=self._loop, name=f"sgn-agent-{self.agent_id}", daemon=True)
        self._loop_thread.start()

    def stop(self):
        if not self._running and self._loop_thread is None:
            return
        self._running = False
        self._stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None
        if self.tcp is not None:
            self.tcp.stop()
        if self.relay is not None:
            self.relay.stop()
        self.chain = None
        self.mesh = None
        self.sync_node = None

    def _loop(self):
        last_heartbeat = 0.0
        while not self._stop.is_set():
            try:
                if self.chain is not None:
                    self.chain.poll(0.05)
                self._replay_outbox()
                now = time.monotonic()
                if now - last_heartbeat >= 1.0:
                    self._send_heartbeat()
                    last_heartbeat = now
            except Exception:
                logger.warning("agent loop error", exc_info=True)
                self._stop.wait(0.1)

    def _send_heartbeat(self):
        payload = {"ts": time.time()}
        if self._own_position is not None:
            x, y, z, frame_id = self._own_position
            payload["position"] = {
                "x": x, "y": y, "z": z,
                "frame": frame_id,
                "confidence": 1.0,
            }
        env = Envelope(msg_id=new_msg_id(), msg_type="heartbeat",
                       sender=self.agent_id, recipient="*",
                       topic=f"/shugunet/{self.agent_id}/heartbeat",
                       payload=payload)
        try:
            self.chain.send(env, qos="best_effort")
        except Exception:
            pass

    def _replay_outbox(self):
        for entry in self.outbox.pending():
            try:
                self.chain.send_raw(entry["peer"], entry["frame"])
            except Exception:
                break

    def _on_envelope(self, env, via):
        with self._lock:
            self._stats["received"] += 1
        if self.on_message is not None:
            try:
                self.on_message(env.sender, {
                    "msg_type": env.msg_type,
                    "topic": env.topic,
                    "payload": env.payload,
                    "via": via,
                })
            except Exception:
                logger.warning("on_message failed", exc_info=True)

    def send(self, peer, topic, payload):
        env = Envelope(msg_id=new_msg_id(), msg_type="task_request",
                       sender=self.agent_id, recipient=peer,
                       topic=topic, payload=payload)
        report = self.chain.send(env, qos="at_least_once")
        with self._lock:
            if report.ok:
                self._stats["sent"] += 1
            else:
                self._stats["errors"] += 1
        return {"status": "success" if report.ok else "failed",
                "via": report.via, "peer": peer}

    def query(self, query, peers=None, top_k=8):
        return self.mesh.query(query, peers=peers, top_k=top_k,
                               poller=lambda: self.chain.poll(0.01) if self.chain else None)

    def sync(self, peer=None, since=None):
        target = peer or "*"
        self.sync_node.send_digests(target)
        return {"status": "success", "peer": target}

    def list_agents(self):
        return []

    def status(self):
        with self._lock:
            stats = dict(self._stats)
        return {
            "agent_id": self.agent_id,
            "shugonet_version": version.VERSION,
            "running": self._running,
            "realm": self.realm,
            "stats": stats,
            "store_count": self.store.count(),
            "outbox_pending": len(self.outbox),
            "chain": self.chain.stats() if self.chain else {},
            "spatial_observations": self.spatial.stats().get("inserts", 0),
            "own_position": self._own_position,
        }

    # -- spatial awareness ----------------------------------------------------

    def set_position(self, x, y, z, frame_id="world"):
        """Set the agent's own estimated position (sent in heartbeats)."""
        self._own_position = (float(x), float(y), float(z), str(frame_id))

    def publish_observation(self, entity_id, x, y, z,
                            confidence=1.0, label="", frame_id="world"):
        """Publish a spatial observation to the fleet."""
        if self.spatial_sync is None:
            return {"status": "refused", "reason": "not connected"}
        from spatial import SpatialObservation
        import time as _t
        obs = SpatialObservation(entity_id=entity_id,
                                 agent_id=self.agent_id,
                                 x=float(x), y=float(y), z=float(z),
                                 confidence=float(confidence),
                                 timestamp=_t.time(),
                                 frame_id=str(frame_id) or "world",
                                 label=str(label))
        ok = self.spatial_sync.publish_observation(obs)
        return {"status": "success" if ok else "failed"}

    def get_fleet_map(self):
        """Return the consolidated spatial view of the fleet."""
        if self.spatial_sync is None:
            return {}
        return self.spatial_sync.get_fleet_map()

    def query_nearby(self, x, y, z, radius):
        """Query spatial observations near a point."""
        if self.spatial_sync is None:
            return []
        return [o.to_dict() for o
                in self.spatial_sync.query_nearby(x, y, z, radius)]

    def status(self):
        with self._lock:
            stats = dict(self._stats)
        return {
            "agent_id": self.agent_id,
            "shugonet_version": version.VERSION,
            "running": self._running,
            "realm": self.realm,
            "stats": stats,
            "store_count": self.store.count(),
            "outbox_pending": len(self.outbox),
            "chain": self.chain.stats() if self.chain else {},
            "spatial_observations": self.spatial.stats().get("inserts", 0),
            "own_position": self._own_position,
        }
