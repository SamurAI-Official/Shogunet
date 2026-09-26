"""
Shogunet NRR (Neural Rendering Runtime) Perception & Render Mesh Adapter
========================================================================

Integrates Shugocore's NRR perception and neural-rendering contract into
Shogunet's multi-agent networking stack.

Zero raw pixel bytes ever cross the mesh:
- Primary agents dispatch lightweight descriptors (`NRRFrameDescriptor` or
  `NRRMotionRequest`).
- Peripheral compute nodes execute neural inference or sensor fusion locally.
- Results return as structured metadata (`NRRRenderResult`, `NRRSceneResult`,
  `NRRSensorEvent`) via validated Shogunet Envelopes.

Capability-aware routing ensures that compute requests are only routed to
nodes that explicitly advertise `compute_caps` with the `nrr_render` workload.
"""

from dataclasses import asdict, dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional

from protocol import Envelope, new_msg_id
from security import sanitize_text

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# Workload identifier for NRR capability matrix
NRR_WORKLOAD = "nrr_render"

# Topics in the /shugunet namespace
TOPIC_NRR_CAPABILITIES = "nrr/capabilities"
TOPIC_NRR_RENDER = "nrr/render"
TOPIC_NRR_RESULT = "nrr/result"
TOPIC_NRR_SCENE_REQUEST = "nrr/scene_request"
TOPIC_NRR_SCENE_RESULT = "nrr/scene_result"
TOPIC_NRR_MOTION_EVENT = "nrr/motion_event"

# Vocabularies aligned with ShugoCore NRR perception contract
KNOWN_PIXEL_FORMATS = ("RGB8", "RGBA8", "YUV420", "GRAY8")
KNOWN_QUALITY_HINTS = ("draft", "balanced", "quality")
KNOWN_UNITS = ("m", "cm", "mm", "pixel", "normalized")
KNOWN_FRAMES = (
    "camera_local",
    "world_aligned",
    "sensor_fused",
    "dh_link",
    "dh_joint",
    "device_chassis",
)
KNOWN_AXES_CONVENTIONS = (
    "cartesian_xyz_right_handed_y_up",
    "cartesian_xyz_right_handed_z_up",
    "pixel_xy_image",
    "dh_standard",
    "spherical_radius_azimuth_elevation",
)
KNOWN_ORIGINS = (
    "camera_optical_center",
    "world_origin",
    "device_chassis_origin",
    "link_base",
    "link_i",
    "joint_i",
    "fused_origin",
)

@dataclass
class Coordinate3D:
    """A single 3-axis measurement with explicit frame + provenance."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    unit: str = "m"
    frame: str = "camera_local"
    axes_convention: str = "cartesian_xyz_right_handed_y_up"
    origin: str = "camera_optical_center"
    sensor_provenance: str = ""
    sensor_device_id: str = ""
    confidence: float = 0.0
    age_ms: int = 0
    source_frame_id: str = ""
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if not isinstance(self.x, (int, float)) or self.x != self.x:
            raise ValueError("x must be a finite number")
        if not isinstance(self.y, (int, float)) or self.y != self.y:
            raise ValueError("y must be a finite number")
        if not isinstance(self.z, (int, float)) or self.z != self.z:
            raise ValueError("z must be a finite number")
        if self.unit not in KNOWN_UNITS:
            raise ValueError("unknown unit " + repr(self.unit))
        if self.frame not in KNOWN_FRAMES:
            raise ValueError("unknown frame " + repr(self.frame))
        if self.axes_convention not in KNOWN_AXES_CONVENTIONS:
            raise ValueError("unknown axes_convention " + repr(self.axes_convention))
        if self.origin not in KNOWN_ORIGINS:
            raise ValueError("unknown origin " + repr(self.origin))
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence out of range 0..1")
        if self.age_ms < 0:
            raise ValueError("age_ms must be >= 0")
        if len(self.sensor_provenance) > 128:
            raise ValueError("sensor_provenance too long")
        if len(self.sensor_device_id) > 128:
            raise ValueError("sensor_device_id too long")
        if len(self.source_frame_id) > 128:
            raise ValueError("source_frame_id too long")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Coordinate3D":
        return Coordinate3D(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
            unit=str(data.get("unit", "m")),
            frame=str(data.get("frame", "camera_local")),
            axes_convention=str(data.get("axes_convention", "cartesian_xyz_right_handed_y_up")),
            origin=str(data.get("origin", "camera_optical_center")),
            sensor_provenance=str(data.get("sensor_provenance", ""))[:128],
            sensor_device_id=str(data.get("sensor_device_id", ""))[:128],
            confidence=float(data.get("confidence", 0.0)),
            age_ms=int(data.get("age_ms", 0)),
            source_frame_id=str(data.get("source_frame_id", ""))[:128],
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class SceneEntity:
    """One detected / reasoned-about thing in the scene."""
    id: str = ""
    label: str = ""
    position: Coordinate3D = field(default_factory=Coordinate3D)
    confidence: float = 0.0
    age_ms: int = 0
    source_frame_id: str = ""
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if not self.id or len(self.id) > 128:
            raise ValueError("entity id required")
        if len(self.label) > 128:
            raise ValueError("label too long")
        self.position.validate()
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence out of range 0..1")
        if self.age_ms < 0:
            raise ValueError("age_ms must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "id": self.id,
            "label": self.label,
            "position": self.position.to_dict(),
            "confidence": self.confidence,
            "age_ms": self.age_ms,
            "source_frame_id": self.source_frame_id,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SceneEntity":
        pos = data.get("position") or {}
        return SceneEntity(
            id=str(data.get("id", ""))[:128],
            label=str(data.get("label", ""))[:128],
            position=Coordinate3D.from_dict(pos),
            confidence=float(data.get("confidence", 0.0)),
            age_ms=int(data.get("age_ms", 0)),
            source_frame_id=str(data.get("source_frame_id", ""))[:128],
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class NRRSensorEvent:
    """One motion / change-detection event from the peripheral worker."""
    event_id: str = ""
    event_type: str = "motion"
    region: Optional[Coordinate3D] = None
    motion_score: float = 0.0
    source_frame_id: str = ""
    previous_frame_id: str = ""
    timestamp_ms: int = 0
    confidence: float = 0.0
    age_ms: int = 0
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if not self.event_id or len(self.event_id) > 128:
            raise ValueError("event_id required")
        if self.event_type not in ("motion", "appearance", "disappearance", "change"):
            raise ValueError("unknown event_type " + repr(self.event_type))
        if self.region is not None:
            self.region.validate()
        if not (0.0 <= self.motion_score <= 1.0):
            raise ValueError("motion_score out of range 0..1")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence out of range 0..1")
        if self.age_ms < 0:
            raise ValueError("age_ms must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "region": self.region.to_dict() if self.region else None,
            "motion_score": self.motion_score,
            "source_frame_id": self.source_frame_id,
            "previous_frame_id": self.previous_frame_id,
            "timestamp_ms": self.timestamp_ms,
            "confidence": self.confidence,
            "age_ms": self.age_ms,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NRRSensorEvent":
        region = data.get("region")
        return NRRSensorEvent(
            event_id=str(data.get("event_id", ""))[:128],
            event_type=str(data.get("event_type", "motion")),
            region=Coordinate3D.from_dict(region) if isinstance(region, dict) else None,
            motion_score=float(data.get("motion_score", 0.0)),
            source_frame_id=str(data.get("source_frame_id", ""))[:128],
            previous_frame_id=str(data.get("previous_frame_id", ""))[:128],
            timestamp_ms=int(data.get("timestamp_ms", 0)),
            confidence=float(data.get("confidence", 0.0)),
            age_ms=int(data.get("age_ms", 0)),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class NRRFrameDescriptor:
    """Lightweight render request -- no pixel bytes, descriptor only."""
    frame_id: str
    width: int
    height: int
    pixel_format: str = "RGB8"
    model_id: str = ""
    reference_id: str = ""
    frame_index: int = 0
    delta_time: float = 0.0
    quality_hint: str = "balanced"
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if not self.frame_id or len(self.frame_id) > 64:
            raise ValueError("frame_id required (<=64 chars)")
        if not (1 <= self.width <= 8192 and 1 <= self.height <= 8192):
            raise ValueError("resolution out of range (1..8192 each axis)")
        if self.pixel_format not in KNOWN_PIXEL_FORMATS:
            raise ValueError("unknown pixel_format " + repr(self.pixel_format))
        if len(self.model_id) > 128:
            raise ValueError("model_id too long (<=128 chars)")
        if len(self.reference_id) > 128:
            raise ValueError("reference_id too long (<=128 chars)")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not (0.0 <= self.delta_time < 3600.0):
            raise ValueError("delta_time out of range")
        if self.quality_hint not in KNOWN_QUALITY_HINTS:
            raise ValueError("unknown quality_hint " + repr(self.quality_hint))

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NRRFrameDescriptor":
        desc = NRRFrameDescriptor(
            frame_id=str(data.get("frame_id", "")),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            pixel_format=str(data.get("pixel_format", "RGB8")),
            model_id=str(data.get("model_id", ""))[:128],
            reference_id=str(data.get("reference_id", ""))[:128],
            frame_index=int(data.get("frame_index", 0)),
            delta_time=float(data.get("delta_time", 0.0)),
            quality_hint=str(data.get("quality_hint", "balanced")),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )
        desc.validate()
        return desc

@dataclass
class NRRMotionRequest:
    """Motion / change-detection request descriptor (peripheral worker)."""
    frame_id: str
    previous_frame_id: str = ""
    motion_threshold: float = 0.15
    min_change_area_pixels: int = 8
    change_detection: bool = True
    motion_detection: bool = True
    max_detections: int = 16
    region_of_interest: Optional[Coordinate3D] = None
    call_nrr_model: bool = True
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if not self.frame_id or len(self.frame_id) > 64:
            raise ValueError("frame_id required (<=64 chars)")
        if len(self.previous_frame_id) > 64:
            raise ValueError("previous_frame_id too long (<=64 chars)")
        if not (0.0 <= self.motion_threshold <= 1.0):
            raise ValueError("motion_threshold out of range 0..1")
        if self.min_change_area_pixels < 1:
            raise ValueError("min_change_area_pixels must be >= 1")
        if self.max_detections < 1 or self.max_detections > 256:
            raise ValueError("max_detections out of range 1..256")
        if self.region_of_interest is not None:
            if not isinstance(self.region_of_interest, Coordinate3D):
                raise ValueError("region_of_interest must be Coordinate3D")
            self.region_of_interest.validate()

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "frame_id": self.frame_id,
            "previous_frame_id": self.previous_frame_id,
            "motion_threshold": self.motion_threshold,
            "min_change_area_pixels": self.min_change_area_pixels,
            "change_detection": self.change_detection,
            "motion_detection": self.motion_detection,
            "max_detections": self.max_detections,
            "region_of_interest": self.region_of_interest.to_dict() if self.region_of_interest else None,
            "call_nrr_model": self.call_nrr_model,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NRRMotionRequest":
        roi = data.get("region_of_interest")
        return NRRMotionRequest(
            frame_id=str(data.get("frame_id", "")),
            previous_frame_id=str(data.get("previous_frame_id", ""))[:64],
            motion_threshold=float(data.get("motion_threshold", 0.15)),
            min_change_area_pixels=int(data.get("min_change_area_pixels", 8)),
            change_detection=bool(data.get("change_detection", True)),
            motion_detection=bool(data.get("motion_detection", True)),
            max_detections=int(data.get("max_detections", 16)),
            region_of_interest=Coordinate3D.from_dict(roi) if isinstance(roi, dict) else None,
            call_nrr_model=bool(data.get("call_nrr_model", True)),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class NRRRenderStats:
    render_time_ms: float = 0.0
    neural_inference_time_ms: float = 0.0
    backend_overhead_ms: float = 0.0
    memory_used_mb: int = 0
    quality_metric: float = 0.0
    backend: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NRRRenderResult:
    frame_id: str
    status: str = "ok"
    output_handle: str = ""
    width: int = 0
    height: int = 0
    stats: NRRRenderStats = field(default_factory=NRRRenderStats)
    error: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def not_supported(frame_id: str, reason: str = "") -> "NRRRenderResult":
        return NRRRenderResult(
            frame_id=frame_id, status="not_supported", error=str(reason)[:256]
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NRRRenderResult":
        stats = data.get("stats") or {}
        return NRRRenderResult(
            frame_id=str(data.get("frame_id", "")),
            status=str(data.get("status", "ok")),
            output_handle=str(data.get("output_handle", "")),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            stats=NRRRenderStats(
                render_time_ms=float(stats.get("render_time_ms", 0.0)),
                neural_inference_time_ms=float(stats.get("neural_inference_time_ms", 0.0)),
                backend_overhead_ms=float(stats.get("backend_overhead_ms", 0.0)),
                memory_used_mb=int(stats.get("memory_used_mb", 0)),
                quality_metric=float(stats.get("quality_metric", 0.0)),
                backend=str(stats.get("backend", "cpu"))[:32],
            ),
            error=str(data.get("error", ""))[:256],
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

@dataclass
class NRRSceneResult:
    frame_id: str
    status: str = "ok"
    source_device_id: str = ""
    sensor_provenance: str = ""
    schema_version: int = SCHEMA_VERSION
    scene_version: int = 1
    entities: List[SceneEntity] = field(default_factory=list)
    motion_events: List[NRRSensorEvent] = field(default_factory=list)
    motion_summary: Dict[str, Any] = field(default_factory=dict)
    model_contribution: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def validate(self) -> None:
        if not self.frame_id or len(self.frame_id) > 128:
            raise ValueError("frame_id required")
        if not self.source_device_id or len(self.source_device_id) > 128:
            raise ValueError("source_device_id required")
        if len(self.sensor_provenance) > 128:
            raise ValueError("sensor_provenance too long")
        for e in self.entities:
            if not isinstance(e, SceneEntity):
                raise ValueError("entities must be SceneEntity")
            e.validate()
        for ev in self.motion_events:
            if not isinstance(ev, NRRSensorEvent):
                raise ValueError("motion_events must be NRRSensorEvent")
            ev.validate()
        if len(self.error) > 256:
            raise ValueError("error too long")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "frame_id": self.frame_id,
            "status": self.status,
            "source_device_id": self.source_device_id,
            "sensor_provenance": self.sensor_provenance,
            "schema_version": self.schema_version,
            "scene_version": self.scene_version,
            "entities": [e.to_dict() for e in self.entities],
            "motion_events": [ev.to_dict() for ev in self.motion_events],
            "motion_summary": dict(self.motion_summary),
            "model_contribution": dict(self.model_contribution),
            "error": self.error,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NRRSceneResult":
        entities = []
        for ed in (data.get("entities") or []):
            if isinstance(ed, dict):
                entities.append(SceneEntity.from_dict(ed))
        events = []
        for ed in (data.get("motion_events") or []):
            if isinstance(ed, dict):
                events.append(NRRSensorEvent.from_dict(ed))
        return NRRSceneResult(
            frame_id=str(data.get("frame_id", "")),
            status=str(data.get("status", "ok")),
            source_device_id=str(data.get("source_device_id", ""))[:128],
            sensor_provenance=str(data.get("sensor_provenance", ""))[:128],
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            scene_version=int(data.get("scene_version", 1)),
            entities=entities,
            motion_events=events,
            motion_summary=dict(data.get("motion_summary") or {}),
            model_contribution=dict(data.get("model_contribution") or {}),
            error=str(data.get("error", ""))[:256],
        )

    @staticmethod
    def not_supported(frame_id: str, reason: str = "") -> "NRRSceneResult":
        return NRRSceneResult(
            frame_id=frame_id, status="not_supported", error=str(reason)[:256]
        )

class NRRMeshAdapter:
    """Network adapter connecting a Shogunet agent to the NRR fleet mesh.

    Integrates with TransportChain and AgentRegistry to route NRR render and
    scene perception requests to capable nodes across 5G/4G/WiFi/LoRa.
    """

    def __init__(self, agent_id: str, chain: Any, registry: Optional[Any] = None,
                 worker: Optional[Any] = None, spatial_sync: Optional[Any] = None,
                 audit: Optional[Any] = None):
        self.agent_id = sanitize_text(agent_id, 48).strip()
        self.chain = chain
        self.registry = registry
        self.worker = worker
        self.spatial_sync = spatial_sync
        self.audit = audit
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
        if self.chain:
            self.chain.subscribe(self._on_envelope)

    def dispatch_render(self, peer: str, descriptor: Dict[str, Any],
                        qos: str = "best_effort") -> Dict[str, Any]:
        """Dispatch a neural render request to a capable peer."""
        try:
            desc = NRRFrameDescriptor.from_dict(descriptor)
            desc.validate()
        except Exception as exc:
            self._audit("nrr_render_invalid_descriptor", {"peer": peer, "error": str(exc)})
            return {"status": "refused", "reason": f"invalid descriptor: {exc}"}

        if self.registry and not self.registry.is_paired(peer):
            self._audit("nrr_render_refused_unpaired", {"peer": peer})
            return {"status": "refused", "reason": "peer not paired"}

        from agent_registry import node_supports_workload
        if self.registry and not node_supports_workload(self.registry.manifest(peer), NRR_WORKLOAD):
            self._audit("nrr_render_refused_no_capability", {"peer": peer})
            return {"status": "refused", "reason": f"peer does not support {NRR_WORKLOAD}"}

        request_id = f"nrr-{int(time.time()*1000)}-{new_msg_id()}"
        topic = f"/shugunet/{self.agent_id}/{TOPIC_NRR_RENDER}"
        payload = {
            "type": "NRRRender",
            "request_id": request_id,
            "descriptor": desc.to_dict(),
        }
        env = Envelope(
            msg_id=new_msg_id(),
            msg_type="nrr_render_request",
            sender=self.agent_id,
            recipient=peer,
            topic=topic,
            payload=payload,
        )
        report = self.chain.send(env, qos=qos)
        self._audit("nrr_render_dispatched", {"peer": peer, "frame_id": desc.frame_id, "ok": report.ok})
        return {"status": "success" if report.ok else "failed", "request_id": request_id, "via": report.via}

    def dispatch_scene_request(self, peer: str, motion_request: Dict[str, Any],
                               qos: str = "best_effort") -> Dict[str, Any]:
        """Dispatch a scene/motion perception request to a capable peer."""
        try:
            req = NRRMotionRequest.from_dict(motion_request)
            req.validate()
        except Exception as exc:
            self._audit("nrr_scene_invalid_request", {"peer": peer, "error": str(exc)})
            return {"status": "refused", "reason": f"invalid motion request: {exc}"}

        if self.registry and not self.registry.is_paired(peer):
            return {"status": "refused", "reason": "peer not paired"}

        from agent_registry import node_supports_workload
        if self.registry and not node_supports_workload(self.registry.manifest(peer), NRR_WORKLOAD):
            return {"status": "refused", "reason": f"peer does not support {NRR_WORKLOAD}"}

        request_id = f"scene-{int(time.time()*1000)}-{new_msg_id()}"
        topic = f"/shugunet/{self.agent_id}/{TOPIC_NRR_SCENE_REQUEST}"
        payload = {
            "type": "NRRSceneRequest",
            "request_id": request_id,
            "motion_request": req.to_dict(),
        }
        env = Envelope(
            msg_id=new_msg_id(),
            msg_type="nrr_scene_request",
            sender=self.agent_id,
            recipient=peer,
            topic=topic,
            payload=payload,
        )
        report = self.chain.send(env, qos=qos)
        self._audit("nrr_scene_dispatched", {"peer": peer, "frame_id": req.frame_id, "ok": report.ok})
        return {"status": "success" if report.ok else "failed", "request_id": request_id, "via": report.via}

    def publish_motion_event(self, event: Dict[str, Any], qos: str = "best_effort") -> Dict[str, Any]:
        """Broadcast a local NRR motion/change event to the fleet mesh."""
        try:
            ev = NRRSensorEvent.from_dict(event)
            ev.validate()
        except Exception as exc:
            return {"status": "refused", "reason": f"invalid event: {exc}"}

        topic = f"/shugunet/{self.agent_id}/{TOPIC_NRR_MOTION_EVENT}"
        payload = {
            "type": "NRRSensorEvent",
            "request_id": ev.event_id,
            "event": ev.to_dict(),
        }
        env = Envelope(
            msg_id=new_msg_id(),
            msg_type="nrr_motion_event",
            sender=self.agent_id,
            recipient="*",
            topic=topic,
            payload=payload,
        )
        report = self.chain.send(env, qos=qos)
        return {"status": "success" if report.ok else "failed", "via": report.via}

    def _on_envelope(self, env: Envelope, via: str) -> None:
        """Handle incoming NRR requests or perception results."""
        if env.sender == self.agent_id:
            return

        if env.msg_type == "nrr_render_request":
            self._handle_inbound_render_request(env)
        elif env.msg_type == "nrr_scene_request":
            self._handle_inbound_scene_request(env)
        elif env.msg_type == "nrr_render_result":
            self._audit("nrr_render_result_received", {"from": env.sender, "frame_id": env.payload.get("frame_id")})
        elif env.msg_type == "nrr_scene_result":
            self._handle_inbound_scene_result(env)
        elif env.msg_type == "nrr_motion_event":
            self._handle_inbound_motion_event(env)

    def _handle_inbound_render_request(self, env: Envelope) -> None:
        """Answer a peer's render request, or fail closed.

        Whatever happens, the requester is owed exactly one reply envelope:
        a request that silently goes unanswered would leave the caller
        blocking on a request_id that can never resolve.
        """
        request_id = env.payload.get("request_id", "")
        descriptor = env.payload.get("descriptor") or {}
        frame_id = descriptor.get("frame_id", "")
        try:
            desc = NRRFrameDescriptor.from_dict(descriptor)
            frame_id = desc.frame_id
        except Exception:
            pass                      # reply with not_supported, not silence
        result_payload = None
        if self.worker is not None and hasattr(self.worker, "render"):
            try:
                result_payload = self.worker.render(env.payload)
            except Exception as exc:
                self._audit("nrr_render_worker_failed",
                            {"from": env.sender, "error": str(exc)[:200]})
                result_payload = None
        if not isinstance(result_payload, dict):
            reason = ("nrr worker unavailable on this peer"
                      if result_payload is None and self.worker is None
                      else "nrr render worker failed on this peer")
            result_payload = {
                "type": "NRRResult", "request_id": request_id,
                "result": NRRRenderResult.not_supported(frame_id,
                                                        reason).to_dict()}

        resp_env = Envelope(
            msg_id=new_msg_id(),
            msg_type="nrr_render_result",
            sender=self.agent_id,
            recipient=env.sender,
            topic=f"/shugunet/{self.agent_id}/{TOPIC_NRR_RESULT}",
            payload=result_payload,
        )
        try:
            self.chain.send(resp_env, qos="best_effort")
        except Exception as exc:
            self._audit("nrr_render_reply_failed",
                        {"to": env.sender, "error": str(exc)[:200]})

    def _handle_inbound_scene_request(self, env: Envelope) -> None:
        """Answer a peer's scene request, or fail closed (never silently)."""
        request_id = env.payload.get("request_id", "")
        req_dict = env.payload.get("motion_request") or {}
        frame_id = req_dict.get("frame_id", "")
        result_payload = None
        if self.worker is not None and hasattr(self.worker, "scene"):
            try:
                result_payload = self.worker.scene(env.payload)
            except Exception as exc:
                self._audit("nrr_scene_worker_failed",
                            {"from": env.sender, "error": str(exc)[:200]})
                result_payload = None
        if not isinstance(result_payload, dict):
            result = NRRSceneResult.not_supported(
                frame_id, "nrr scene worker unavailable on this peer")
            result_payload = {"type": "NRRSceneResult",
                              "request_id": request_id,
                              "result": result.to_dict()}
        resp_env = Envelope(
            msg_id=new_msg_id(),
            msg_type="nrr_scene_result",
            sender=self.agent_id,
            recipient=env.sender,
            topic=f"/shugunet/{self.agent_id}/{TOPIC_NRR_SCENE_RESULT}",
            payload=result_payload,
        )
        try:
            self.chain.send(resp_env, qos="best_effort")
        except Exception as exc:
            self._audit("nrr_scene_reply_failed",
                        {"to": env.sender, "error": str(exc)[:200]})

    def _handle_inbound_scene_result(self, env: Envelope) -> None:
        """Incorporate received scene entities into local spatial index if present."""
        body = env.payload.get("result") or {}
        try:
            scene = NRRSceneResult.from_dict(body)
            if self.spatial_sync is not None and scene.status == "ok":
                from spatial import SpatialObservation
                now = time.time()
                for ent in scene.entities:
                    obs = SpatialObservation(
                        entity_id=ent.id,
                        agent_id=env.sender,
                        x=ent.position.x,
                        y=ent.position.y,
                        z=ent.position.z,
                        confidence=ent.confidence,
                        timestamp=now,
                        frame_id=ent.position.frame,
                        label=ent.label,
                    )
                    self.spatial_sync.index.insert(obs)
        except Exception as exc:
            logger.warning("failed to integrate inbound scene result: %s", exc)

    def _handle_inbound_motion_event(self, env: Envelope) -> None:
        """Process peer motion event."""
        body = env.payload.get("event") or {}
        try:
            ev = NRRSensorEvent.from_dict(body)
            if self.spatial_sync is not None and ev.region is not None:
                from spatial import SpatialObservation
                obs = SpatialObservation(
                    entity_id=f"motion-{ev.event_id}",
                    agent_id=env.sender,
                    x=ev.region.x,
                    y=ev.region.y,
                    z=ev.region.z,
                    confidence=ev.confidence,
                    timestamp=time.time(),
                    frame_id=ev.region.frame,
                    label=f"motion_{ev.event_type}",
                )
                self.spatial_sync.index.insert(obs)
        except Exception as exc:
            logger.warning("failed to integrate inbound motion event: %s", exc)

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.audit:
            try:
                self.audit.append(event_type, payload)
            except Exception:
                pass

