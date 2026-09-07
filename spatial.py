"""
Shogunet 3D spatial awareness
==============================

Core data types and an octree-based spatial index for tracking entities
across a multi-agent fleet. Agents publish observations of entities
(objects / other agents / obstacles) with 3D coordinates; the index fuses
overlapping observations from multiple agents into a single consolidated
view of the environment.

Key concepts
------------
- **SpatialObservation**: one agent's observation of one entity at a
  3D point in a named coordinate frame.
- **SpatialIndex**: octree-backed store with spatial predicates (sphere,
  AABB, agent, entity), confidence-weighted fusion, and temporal decay.
- **Coordinate frames**: "world" is the shared reference; agents may also
  publish in "agent-xxx-local" frames.  The fleet host maintains frame
  transforms when agents share them via ``coordinate_frame`` messages.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from security import sanitize_text

logger = __import__("logging").getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_OBS_LABEL = 64
MAX_OBS_ENTITY_ID = 64
MAX_OBS_FRAME_ID = 48
DEFAULT_OCTREE_DEPTH = 8
DEFAULT_LEAF_SIZE = 1.0
MAX_OBS_PER_LEAF = 1000
DEFAULT_TEMPORAL_DECAY_S = 60.0
MIN_CONFIDENCE = 0.001
FUSION_MIN_AGREEMENT = 2
# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SpatialObservation:
    entity_id: str
    agent_id: str
    x: float
    y: float
    z: float
    confidence: float
    timestamp: float
    frame_id: str = "world"
    label: str = ""
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.entity_id = sanitize_text(self.entity_id, MAX_OBS_ENTITY_ID)
        self.agent_id = sanitize_text(self.agent_id, 48)
        self.frame_id = sanitize_text(self.frame_id, MAX_OBS_FRAME_ID) or "world"
        self.label = sanitize_text(self.label, MAX_OBS_LABEL)
        self.confidence = max(MIN_CONFIDENCE, min(1.0, float(self.confidence)))
        self.timestamp = float(self.timestamp) if self.timestamp else time.time()
        if not self.entity_id:
            raise ValueError("entity_id is required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id, "agent_id": self.agent_id,
            "x": self.x, "y": self.y, "z": self.z,
            "confidence": self.confidence, "timestamp": self.timestamp,
            "frame_id": self.frame_id, "label": self.label,
            "orientation": list(self.orientation),
            "size": list(self.size),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpatialObservation":
        return cls(
            entity_id=str(d.get("entity_id", "")),
            agent_id=str(d.get("agent_id", "")),
            x=float(d.get("x", 0.0)), y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            confidence=float(d.get("confidence", 1.0)),
            timestamp=float(d.get("timestamp", 0.0)),
            frame_id=str(d.get("frame_id", "world")),
            label=str(d.get("label", "")),
            orientation=tuple(d.get("orientation", (0.0, 0.0, 0.0, 1.0))),
            size=tuple(d.get("size", (0.0, 0.0, 0.0))),
            metadata=dict(d.get("metadata", {})),
        )

    def age(self, now: Optional[float] = None) -> float:
        return (now or time.time()) - self.timestamp

    def effective_confidence(self, now=None,
                             decay_s=DEFAULT_TEMPORAL_DECAY_S) -> float:
        a = self.age(now)
        if a <= decay_s:
            return self.confidence
        factor = max(0.0, 1.0 - (a - decay_s) / decay_s)
# ---------------------------------------------------------------------------
# Octree spatial index
# ---------------------------------------------------------------------------

@dataclass
class FrameTransform:
    """A known transform from ``from_frame`` to ``to_frame``."""
    from_frame: str
    to_frame: str
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    confidence: float = 1.0

    def apply(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return x - self.tx, y - self.ty, z - self.tz

class _OctreeNode:
    """One node in the spatial octree."""

    __slots__ = ("cx", "cy", "cz", "half", "obs", "children", "size")

    def __init__(self, cx: float, cy: float, cz: float, half: float):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.half = half
        self.obs: Dict[str, List[SpatialObservation]] = {}
        self.children: Optional[List["_OctreeNode"]] = None
        self.size = 0

    def _contains(self, x: float, y: float, z: float) -> bool:
        return (abs(self.cx - x) <= self.half and
                abs(self.cy - y) <= self.half and
                abs(self.cz - z) <= self.half)

    def _subdivide(self) -> None:
        h = self.half / 2.0
        off = self.half / 2.0
        self.children = []
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    self.children.append(_OctreeNode(
                        self.cx + dx * off, self.cy + dy * off,
                        self.cz + dz * off, h))

    def insert(self, obs: SpatialObservation, depth=0,
               max_depth=DEFAULT_OCTREE_DEPTH,
               max_per_leaf=MAX_OBS_PER_LEAF) -> None:
        if not self._contains(obs.x, obs.y, obs.z):
            return
        if self.children is None:
            if depth >= max_depth or len(self.obs) < max_per_leaf:
                self.obs.setdefault(obs.entity_id, []).append(obs)
                self.size += 1
                return
            self._subdivide()
        for child in self.children:
            child.insert(obs, depth + 1, max_depth, max_per_leaf)

    def query_sphere(self, x: float, y: float, z: float,
                     radius: float, out: List[SpatialObservation]) -> None:
        if not self._overlaps_sphere(x, y, z, radius):
            return
        for obs_list in self.obs.values():
            for o in obs_list:
                dx, dy, dz = o.x - x, o.y - y, o.z - z
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    out.append(o)
        if self.children:
            for child in self.children:
                child.query_sphere(x, y, z, radius, out)

    def _overlaps_sphere(self, x: float, y: float, z: float,
                         radius: float) -> bool:
        d = 0.0
        for v, c, h in ((x, self.cx, self.half),
                         (y, self.cy, self.half),
                         (z, self.cz, self.half)):
            if v < c - h:
                d += (v - (c - h)) ** 2
            elif v > c + h:
                d += (v - (c + h)) ** 2
        return d <= radius * radius

    def collect_all(self, out: List[SpatialObservation]) -> None:
        for obs_list in self.obs.values():
            out.extend(obs_list)
        if self.children:
            for child in self.children:
                child.collect_all(out)
class SpatialIndex:
    """Octree-backed 3D spatial index for multi-agent observation fusion.

    Thread-safe for concurrent insert and query (coarse RLock).
    """

    def __init__(self, world_size: float = 1000.0,
                 max_depth=DEFAULT_OCTREE_DEPTH,
                 max_per_leaf=MAX_OBS_PER_LEAF,
                 decay_s=DEFAULT_TEMPORAL_DECAY_S):
        self._root = _OctreeNode(0.0, 0.0, 0.0, world_size / 2.0)
        self._max_depth = max(3, int(max_depth))
        self._max_per_leaf = max(1, int(max_per_leaf))
        self._decay_s = max(0.1, float(decay_s))
        self._transforms: Dict[str, FrameTransform] = {}
        self._lock = __import__("threading").RLock()
        self._stats = {"inserts": 0, "queries": 0, "fusions": 0,
                       "removed_expired": 0}

    # -- insert ---------------------------------------------------------------

    def insert(self, obs: SpatialObservation) -> None:
        with self._lock:
            self._root.insert(obs, max_depth=self._max_depth,
                              max_per_leaf=self._max_per_leaf)
            self._stats["inserts"] += 1

    def insert_dict(self, d: Dict[str, Any]) -> None:
        self.insert(SpatialObservation.from_dict(d))

    def add_transform(self, t: FrameTransform) -> None:
        with self._lock:
            k = f"{t.from_frame}\u2192{t.to_frame}"
            self._transforms[k] = t
            rev = FrameTransform(
                from_frame=t.to_frame, to_frame=t.from_frame,
                tx=-t.tx, ty=-t.ty, tz=-t.tz,
                qx=t.qx, qy=t.qy, qz=t.qz, qw=t.qw,
                confidence=t.confidence)
            self._transforms[f"{t.to_frame}\u2192{t.from_frame}"] = rev

    # -- queries --------------------------------------------------------------

    def query_sphere(self, x: float, y: float, z: float,
                     radius: float) -> List[SpatialObservation]:
        with self._lock:
            out: List[SpatialObservation] = []
            self._root.query_sphere(x, y, z, radius, out)
            self._stats["queries"] += 1
            return out

    def query_aabb(self, x1, y1, z1, x2, y2, z2) -> List[SpatialObservation]:
        cx, cy, cz = (x1 + x2) / 2.0, (y1 + y2) / 2.0, (z1 + z2) / 2.0
        r = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) / 2.0
        candidates = self.query_sphere(cx, cy, cz, r)
        return [o for o in candidates
                if x1 <= o.x <= x2 and y1 <= o.y <= y2 and z1 <= o.z <= z2]

    def query_entity(self, entity_id: str) -> List[SpatialObservation]:
        eid = sanitize_text(entity_id, MAX_OBS_ENTITY_ID)
        return [o for o in self._all() if o.entity_id == eid]

    def query_agent(self, agent_id: str) -> List[SpatialObservation]:
        aid = sanitize_text(agent_id, 48)
        return [o for o in self._all() if o.agent_id == aid]

    def query_nearby(self, x, y, z, radius) -> List[SpatialObservation]:
        return self.query_sphere(x, y, z, radius)

    def query_since(self, since: float) -> List[SpatialObservation]:
        return [o for o in self._all() if o.timestamp >= since]

    def all_observations(self) -> List[SpatialObservation]:
        return self._all()

    def agent_positions(self, now=None) -> Dict[str, SpatialObservation]:
        by_agent: Dict[str, List[SpatialObservation]] = {}
        for o in self._all():
            by_agent.setdefault(o.agent_id, []).append(o)
        result = {}
        for aid, entries in by_agent.items():
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            result[aid] = entries[0]
        return result

    def entity_positions(self, now=None) -> Dict[str, SpatialObservation]:
        by_entity: Dict[str, List[SpatialObservation]] = {}
        for o in self._all():
            by_entity.setdefault(o.entity_id, []).append(o)
        result = {}
        for eid, entries in by_entity.items():
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            result[eid] = self._fuse(entries, now)
        return result

    # -- fusion ---------------------------------------------------------------

    def fuse_entity(self, entity_id: str, now=None
                    ) -> Optional[SpatialObservation]:
        obs_list = self.query_entity(entity_id)
        if not obs_list:
            return None
        return self._fuse(obs_list, now)

    def fuse_all(self, now=None) -> Dict[str, SpatialObservation]:
        by_entity: Dict[str, List[SpatialObservation]] = {}
        for o in self._all():
            by_entity.setdefault(o.entity_id, []).append(o)
        result = {}
        for eid, entries in by_entity.items():
            fused = self._fuse(entries, now)
            if fused is not None:
                result[eid] = fused
        return result

    # -- maintenance ----------------------------------------------------------

    def prune_expired(self, max_age_s=300.0, now=None) -> int:
        n = time.time() if now is None else now
        cutoff = n - max_age_s
        removed = 0
        with self._lock:
            all_obs = self._all()
            fresh = [o for o in all_obs if o.timestamp >= cutoff]
            self._rebuild(fresh)
            removed = len(all_obs) - len(fresh)
            self._stats["removed_expired"] += removed
        return removed

    # -- serialisation --------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        with self._lock:
            obs = [o.to_dict() for o in self._all()]
            return {"observations": obs,
                    "stats": dict(self._stats)}

    def deserialize(self, data: Dict[str, Any]) -> None:
        for raw in data.get("observations", []):
            self.insert_dict(raw)

    # -- internals ------------------------------------------------------------

    def _all(self) -> List[SpatialObservation]:
        out: List[SpatialObservation] = []
        self._root.collect_all(out)
        return out

    def _rebuild(self, obs: List[SpatialObservation]) -> None:
        self._root = _OctreeNode(0.0, 0.0, 0.0, self._root.half)
        for o in obs:
            self._root.insert(o, max_depth=self._max_depth,
                              max_per_leaf=self._max_per_leaf)

    def _fuse(self, entries: List[SpatialObservation], now=None
              ) -> Optional[SpatialObservation]:
        now = now or time.time()
        valid = [e for e in entries
                 if e.effective_confidence(now, self._decay_s) > 0]
        if not valid:
            return None
        agents = set(e.agent_id for e in valid)
        if len(valid) < FUSION_MIN_AGREEMENT and len(agents) < FUSION_MIN_AGREEMENT:
            return max(valid, key=lambda e: e.effective_confidence(now, self._decay_s))
        w_sum = sum(e.effective_confidence(now, self._decay_s) for e in valid)
        if w_sum == 0:
            return None
        best = max(valid, key=lambda e: e.effective_confidence(now, self._decay_s))
        fx = sum(e.x * e.effective_confidence(now, self._decay_s) for e in valid) / w_sum
        fy = sum(e.y * e.effective_confidence(now, self._decay_s) for e in valid) / w_sum
        fz = sum(e.z * e.effective_confidence(now, self._decay_s) for e in valid) / w_sum
        fused = SpatialObservation(
            entity_id=best.entity_id, agent_id="fused",
            x=fx, y=fy, z=fz,
            confidence=min(1.0, w_sum / len(valid)),
            timestamp=now, frame_id=best.frame_id, label=best.label,
            orientation=best.orientation, size=best.size,
            metadata={"fused_from": len(valid),
                      "fusing_agents": list(agents)},
        )
        self._stats["fusions"] += 1
        return fused

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)
        return [o for o in self._all() if o.timestamp >= since]

    def all_observations(self) -> List[SpatialObservation]:
        return self._all()

    def agent_positions(self, now=None) -> Dict[str, SpatialObservation]:
        by_agent: Dict[str, List[SpatialObservation]] = {}
        for o in self._all():
            by_agent.setdefault(o.agent_id, []).append(o)
        result = {}
        for aid, entries in by_agent.items():
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            result[aid] = entries[0]
        return result

    def entity_positions(self, now=None) -> Dict[str, SpatialObservation]:
        by_entity: Dict[str, List[SpatialObservation]] = {}
        for o in self._all():
            by_entity.setdefault(o.entity_id, []).append(o)
        result = {}
        for eid, entries in by_entity.items():
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            result[eid] = self._fuse(entries, now)
        return result
        return self.confidence * factor


@dataclass
class FrameTransform:
    from_frame: str
    to_frame: str
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    confidence: float = 1.0

    def apply(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return x - self.tx, y - self.ty, z - self.tz