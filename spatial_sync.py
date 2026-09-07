"""
Shogunet spatial memory sync
==============================

Cross-agent spatial observation sharing and fusion. ``SpatialMemoryNode``
attaches to a TransportChain and is the spatial counterpart of
MemorySyncNode.

- publish_observation: share one spatial observation with the fleet.
- query_spatial: fan-out a spatial sphere query to peers.
- get_fleet_map: consolidated spatial view after fusion.
- auto-answer: respond to spatial queries with local observations.
"""

import logging
import threading
import time as _time
from typing import Any, Callable, Dict, List, Optional

import protocol
from protocol import Envelope
from security import sanitize_text
from spatial import (SpatialIndex, SpatialObservation, FrameTransform)

logger = logging.getLogger(__name__)

DEFAULT_QUERY_TIMEOUT_S = 2.0
MAX_SPATIAL_RESULTS = 128
MERGE_INTERVAL_S = 30.0
MAX_OBS_PER_MERGE = 256


class SpatialMemoryNode:
    """Attaches a spatial awareness layer to a TransportChain."""

    def __init__(self, agent_id, chain, index, profiles=None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        if not self.agent_id:
            raise ValueError("agent_id required")
        self.chain = chain
        self.index = index
        self._profiles = dict(profiles or {})
        self._lock = threading.RLock()
        self._pending = {}
        self._stats = {
            "observations_sent": 0, "observations_received": 0,
            "queries_sent": 0, "queries_answered": 0,
            "queries_replied": 0, "merges_sent": 0,
        }
        self._last_merge_ts = 0.0
        chain.subscribe(self._on_envelope)

    # -- publish ---------------------------------------------------------------

    def publish_observation(self, obs, qos="at_least_once"):
        self.index.insert(obs)
        env = Envelope(msg_id=protocol.new_msg_id(),
                       msg_type="spatial_observation",
                       sender=self.agent_id, recipient="*",
                       topic=f"/shugunet/{self.agent_id}/spatial",
                       payload=obs.to_dict())
        ok = self.chain.send(env, qos=qos).ok
        if ok:
            self._stats["observations_sent"] += 1
        return ok

    def publish_position(self, x, y, z, frame_id="world", confidence=1.0):
        obs = SpatialObservation(
            entity_id=self.agent_id, agent_id=self.agent_id,
            x=x, y=y, z=z, confidence=confidence,
            timestamp=_time.time(), frame_id=frame_id, label="agent")
        return self.publish_observation(obs)

    def publish_transform(self, t):
        self.index.add_transform(t)
        env = Envelope(msg_id=protocol.new_msg_id(),
                       msg_type="coordinate_frame",
                       sender=self.agent_id, recipient="*",
                       topic=f"/shugunet/{self.agent_id}/spatial",
                       payload={"from_frame": t.from_frame,
                                "to_frame": t.to_frame,
                                "tx": t.tx, "ty": t.ty, "tz": t.tz,
                                "qx": t.qx, "qy": t.qy, "qz": t.qz,
                                "qw": t.qw, "confidence": t.confidence})
        return self.chain.send(env).ok

    # -- query ----------------------------------------------------------------

    def query_sphere(self, x, y, z, radius, timeout_s=2.0, poller=None):
        if poller is None:
            poller = lambda: self.chain.poll(0.01)
        correlation = protocol.new_msg_id()
        with self._lock:
            self._pending[correlation] = {
                "results": [], "keys": set(), "replies": 0, "expected": 1}
        payload = {"correlation": correlation, "x": x, "y": y,
                   "z": z, "radius": radius, "top_k": 128}
        env = Envelope(msg_id=protocol.new_msg_id(),
                       msg_type="spatial_query",
                       sender=self.agent_id, recipient="*",
                       topic=f"/shugunet/{self.agent_id}/spatial",
                       payload=payload)
        self.chain.send(env, qos="best_effort")
        self._stats["queries_sent"] += 1
        deadline = _time.time() + max(0.1, float(timeout_s))
        while _time.time() < deadline:
            if poller:
                poller()
            with self._lock:
                p = self._pending.get(correlation, {})
                if p.get("replies", 0) >= 1:
                    break
            _time.sleep(0.01)
        with self._lock:
            return list(self._pending.pop(correlation, {}).get("results", []))

    def query_nearby(self, x, y, z, radius):
        return self.index.query_sphere(x, y, z, radius)

    def get_fleet_map(self, now=None):
        now = now or __import__("time").time()
        return {
            "entities": {eid: obs.to_dict()
                         for eid, obs in self.index.fuse_all(now).items()},
            "agents": {aid: obs.to_dict()
                       for aid, obs in self.index.agent_positions(now).items()},
            "observation_count": len(self.index.all_observations()),
            "timestamp": now,
        }

    def get_agent_position(self, agent_id):
        return self.index.agent_positions().get(agent_id)

    # -- periodic merge -------------------------------------------------------

    def send_merge(self):
        now = __import__("time").time()
        fused = self.index.fuse_all(now)
        entries = [obs.to_dict() for obs in fused.values()][:256]
        if not entries:
            return False
        env = Envelope(msg_id=protocol.new_msg_id(),
                       msg_type="spatial_merge",
                       sender=self.agent_id, recipient="*",
                       topic=f"/shugunet/{self.agent_id}/spatial",
                       payload={"observations": entries, "timestamp": now})
        ok = self.chain.send(env).ok
        if ok:
            self._stats["merges_sent"] += 1
            self._last_merge_ts = now
        return ok

    # -- envelope handlers ----------------------------------------------------

    def _on_envelope(self, env, via):
        if env.recipient not in ("*", self.agent_id):
            return
        if env.msg_type == "spatial_observation":
            self._handle_observation(env)
        elif env.msg_type == "spatial_query":
            self._handle_query(env)
        elif env.msg_type == "spatial_response":
            self._handle_response(env)
        elif env.msg_type == "spatial_merge":
            self._handle_merge(env)
        elif env.msg_type == "coordinate_frame":
            self._handle_frame(env)

    def _handle_observation(self, env):
        raw = env.payload
        if not isinstance(raw, dict):
            return
        try:
            obs = SpatialObservation.from_dict(raw)
        except Exception:
            return
        self.index.insert(obs)
        self._stats["observations_received"] += 1

    def _handle_query(self, env):
        if env.sender == self.agent_id:
            return
        payload = env.payload
        if not isinstance(payload, dict):
            return
        x = float(payload.get("x", 0.0))
        y = float(payload.get("y", 0.0))
        z = float(payload.get("z", 0.0))
        radius = float(payload.get("radius", 10.0))
        top_k = int(payload.get("top_k", 128))
        results = self.index.query_sphere(x, y, z, radius)
        results.sort(key=lambda o: o.confidence, reverse=True)
        results = results[:max(1, top_k)]
        correlation = payload.get("correlation")
        reply = Envelope(msg_id=protocol.new_msg_id(),
                         msg_type="spatial_response",
                         sender=self.agent_id, recipient=env.sender,
                         topic=f"/shugunet/{self.agent_id}/spatial",
                         payload={"correlation": correlation,
                                  "results": [r.to_dict() for r in results]})
        if self.chain.send(reply).ok:
            self._stats["queries_answered"] += 1

    def _handle_response(self, env):
        payload = env.payload
        if not isinstance(payload, dict):
            return
        correlation = payload.get("correlation")
        if not isinstance(correlation, int):
            return
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            return
        with self._lock:
            pending = self._pending.get(correlation)
            if pending is None:
                return
            for raw in raw_results:
                try:
                    obs = SpatialObservation.from_dict(raw)
                    key = f"{obs.entity_id}:{obs.agent_id}"
                    if key not in pending["keys"]:
                        pending["keys"].add(key)
                        pending["results"].append(obs)
                except Exception:
                    pass
            pending["replies"] += 1
        self._stats["queries_replied"] += 1

    def _handle_merge(self, env):
        payload = env.payload
        if not isinstance(payload, dict):
            return
        for raw in payload.get("observations", []):
            try:
                obs = SpatialObservation.from_dict(raw)
                self.index.insert(obs)
            except Exception:
                pass

    def _handle_frame(self, env):
        payload = env.payload
        if not isinstance(payload, dict):
            return
        t = FrameTransform(
            from_frame=str(payload.get("from_frame", "")),
            to_frame=str(payload.get("to_frame", "")),
            tx=float(payload.get("tx", 0.0)),
            ty=float(payload.get("ty", 0.0)),
            tz=float(payload.get("tz", 0.0)),
            qx=float(payload.get("qx", 0.0)),
            qy=float(payload.get("qy", 0.0)),
            qz=float(payload.get("qz", 0.0)),
            qw=float(payload.get("qw", 1.0)),
            confidence=float(payload.get("confidence", 1.0)),
        )
        self.index.add_transform(t)

    def maybe_merge(self, interval_s=30.0):
        if _time.time() - self._last_merge_ts >= interval_s:
            return self.send_merge()
        return False

    def stats(self):
        with self._lock:
            return dict(self._stats)