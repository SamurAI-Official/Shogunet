"""NRR (Neural Rendering Runtime) mesh integration tests.

Covers the three seams the NRR feature rests on:

- **Capability routing** -- ``AgentRegistry`` only offers compute work to
  nodes that advertise it, and ``sanitize_compute_caps`` refuses to let a
  peer smuggle unknown workloads or absurd resource claims onto the fleet.
- **Zero-pixel transfer** -- a render request crosses the mesh carrying only
  a descriptor, and the reply carries only a handle plus stats. No envelope
  anywhere in the exchange may contain raw pixel bytes.
- **Dispatch + worker hooks** -- requests are refused for unpaired or
  incapable peers, and a node with no local worker answers
  ``not_supported`` rather than inventing a frame.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_registry import (AgentRegistry, node_supports_workload,
                            sanitize_compute_caps)
from nrr_adapter import (Coordinate3D, NRRFrameDescriptor, NRRMeshAdapter,
                         NRRMotionRequest, NRRRenderResult, NRRSceneResult,
                         SceneEntity)
from protocol import Envelope, new_msg_id
from spatial import SpatialIndex
from spatial_sync import SpatialMemoryNode
from transports import LoopbackBus, LoopbackTransport
from transport_fallback import TransportChain

# Keys that would mean raw pixels/frames leaked onto the mesh.
PIXEL_KEYS = {"pixels", "data", "rgba", "rgb", "frame_bytes", "image",
              "buffer", "b64", "bytes", "framebuffer"}


def pump(*chains, rounds=30):
    for _ in range(rounds):
        for c in chains:
            c.poll(0.01)
        time.sleep(0.005)


def descriptor(frame_id="frame-1", width=320, height=240):
    return NRRFrameDescriptor(frame_id=frame_id, width=width, height=height,
                              model_id="nrc-small",
                              quality_hint="draft").to_dict()


def envelope(msg_type, sender, payload, topic_tail):
    return Envelope(msg_id=new_msg_id(), msg_type=msg_type, sender=sender,
                    recipient="agent-a",
                    topic=f"/shugunet/{sender}/{topic_tail}", payload=payload)


# -- capability matrix -------------------------------------------------------

class TestComputeCaps(unittest.TestCase):

    def test_unknown_workloads_and_values_are_dropped(self):
        caps = sanitize_compute_caps({"compute_caps": {
            "fp16": True, "int8": "yes", "vram_mb": 8_000_000,
            "workloads": ["nrr_render", "vision", "crypto-miner", ""]}})
        self.assertTrue(caps["fp16"])
        self.assertNotIn("int8", caps)             # wrong type
        self.assertNotIn("vram_mb", caps)          # absurd claim
        self.assertEqual(caps["workloads"], ["nrr_render", "vision"])

    def test_non_dict_caps_yield_nothing(self):
        self.assertEqual(sanitize_compute_caps({"compute_caps": "fast"}), {})
        self.assertEqual(sanitize_compute_caps({}), {})

    def test_node_supports_workload(self):
        man = {"compute_caps": {"workloads": ["vision"]}}
        self.assertTrue(node_supports_workload(man, "vision"))
        self.assertFalse(node_supports_workload(man, "nrr_render"))
        self.assertFalse(node_supports_workload({}, "vision"))


class TestCapabilityRouting(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()
        self.registry.pair("gpu-1", manifest={"compute_caps": {
            "workloads": ["nrr_render"], "vram_mb": 8192, "fp16": True}})
        self.registry.pair("cam-1", manifest={"compute_caps": {
            "workloads": ["vision"]}})
        self.registry.pair("brain-1")               # no compute at all

    def test_only_capable_alive_nodes_are_offered(self):
        nodes = self.registry.nodes_for_workload("nrr_render")
        self.assertEqual([n["agent_id"] for n in nodes], ["gpu-1"])

    def test_stale_but_paired_node_is_not_offered(self):
        with self.registry._lock:
            self.registry._last_heartbeat["gpu-1"] = (
                time.monotonic() - self.registry.heartbeat_timeout_s - 1.0)
        self.assertEqual(self.registry.nodes_for_workload("nrr_render"), [])



# -- dispatch ---------------------------------------------------------------

class TestNRRDispatch(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()
        self.registry.pair("gpu-1", manifest={"compute_caps": {
            "workloads": ["nrr_render"]}})
        self.bus = LoopbackBus("nrr-dispatch")
        self.tx_a = LoopbackTransport("agent-a", self.bus)
        # A directed send needs a member to land on, so the peer joins too.
        self.tx_gpu = LoopbackTransport("gpu-1", self.bus)
        self.chain = TransportChain("agent-a", [self.tx_a])
        self.adapter = NRRMeshAdapter("agent-a", self.chain,
                                      registry=self.registry)
        self.peer_saw = []
        self.tx_gpu.subscribe(
            lambda env, via: self.peer_saw.append(env))

    def test_valid_descriptor_dispatches(self):
        res = self.adapter.dispatch_render("gpu-1", descriptor())
        self.assertEqual(res["status"], "success")
        self.assertIn("request_id", res)

    def test_unpaired_peer_refused(self):
        res = self.adapter.dispatch_render("ghost", descriptor())
        self.assertEqual(res["status"], "refused")
        self.assertIn("not paired", res["reason"])

    def test_incapable_peer_refused(self):
        self.registry.pair("brain-1")
        res = self.adapter.dispatch_render("brain-1", descriptor())
        self.assertEqual(res["status"], "refused")
        self.assertIn("nrr_render", res["reason"])

    def test_invalid_descriptor_refused_before_send(self):
        for bad in ({"frame_id": "", "width": 320, "height": 240},
                    {"frame_id": "f", "width": 0, "height": 240},
                    {"frame_id": "f", "width": 320, "height": 240,
                     "pixel_format": "HDR_FLOAT"}):
            res = self.adapter.dispatch_render("gpu-1", bad)
            self.assertEqual(res["status"], "refused", bad)
            self.assertIn("invalid descriptor", res["reason"])
        self.assertEqual(
            [e for e in self.peer_saw if e.msg_type == "nrr_render_request"],
            [], "an invalid descriptor must never reach the wire")

    def test_no_registry_dispatches_without_capability_gate(self):
        loose = NRRMeshAdapter("agent-a", self.chain)
        self.assertEqual(loose.dispatch_render("gpu-1", descriptor())["status"],
                         "success")

    def test_scene_request_dispatch_and_validation(self):
        ok = self.adapter.dispatch_scene_request(
            "gpu-1", {"frame_id": "f1", "previous_frame_id": "f0",
                      "motion_threshold": 0.2})
        self.assertEqual(ok["status"], "success")
        for bad in ({"frame_id": ""},
                    {"frame_id": "f1", "motion_threshold": 5.0},
                    {"frame_id": "f1", "max_detections": 9999}):
            res = self.adapter.dispatch_scene_request("gpu-1", bad)
            self.assertEqual(res["status"], "refused", bad)
            self.assertIn("invalid motion request", res["reason"])


# -- zero-pixel guarantee ---------------------------------------------------

class TestZeroPixelTransfer(unittest.TestCase):
    """The whole point of the NRR mesh: no raw pixels ever cross the wire."""

    def setUp(self):
        self.registry = AgentRegistry()
        self.registry.pair("gpu-1", manifest={"compute_caps": {
            "workloads": ["nrr_render"]}})
        self.bus = LoopbackBus("nrr-pixels")
        self.chain_a = TransportChain(
            "agent-a", [LoopbackTransport("agent-a", self.bus)])
        self.chain_g = TransportChain(
            "gpu-1", [LoopbackTransport("gpu-1", self.bus)])
        self.adapter_a = NRRMeshAdapter("agent-a", self.chain_a,
                                        registry=self.registry)
        self.adapter_g = NRRMeshAdapter("gpu-1", self.chain_g,
                                        registry=self.registry)
        # Capture every NRR envelope in both directions.
        self.wire = []
        for chain in (self.chain_a, self.chain_g):
            chain.subscribe(lambda env, via: self.wire.append(env)
                            if env.msg_type.startswith("nrr_") else None)

    def _results(self):
        return [e for e in self.wire if e.msg_type == "nrr_render_result"]

    def test_no_worker_answers_not_supported_without_pixels(self):
        self.adapter_a.dispatch_render("gpu-1", descriptor())
        pump(self.chain_a, self.chain_g)
        results = self._results()
        self.assertEqual(len(results), 1)
        result = NRRRenderResult.from_dict(results[0].payload["result"])
        self.assertEqual(result.status, "not_supported")
        self.assertEqual(result.output_handle, "")
        self.assertEqual(set(result.to_dict()) & PIXEL_KEYS, set())

    def test_no_envelope_carries_pixel_bytes(self):
        self.adapter_a.dispatch_render("gpu-1", descriptor())
        pump(self.chain_a, self.chain_g)
        self.assertTrue(self.wire)
        for env in self.wire:
            self.assertEqual(set(env.payload) & PIXEL_KEYS, set(),
                             f"{env.msg_type} leaked pixel bytes")

    def test_worker_hook_is_used_when_present(self):
        seen = {}

        class Worker:
            backend = "cpu"
            available = True

            def render(self, request):
                seen["request"] = request
                return {"type": "NRRResult",
                        "request_id": request.get("request_id"),
                        "result": NRRRenderResult(
                            frame_id=request["descriptor"]["frame_id"],
                            status="ok", output_handle="local://frame-1",
                            width=320, height=240).to_dict()}

        self.adapter_g.worker = Worker()
        self.adapter_a.dispatch_render("gpu-1", descriptor())
        pump(self.chain_a, self.chain_g)
        self.assertIn("request", seen)
        result = NRRRenderResult.from_dict(
            self._results()[0].payload["result"])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output_handle, "local://frame-1")

    def test_raising_worker_still_owes_a_reply(self):
        class Boom:
            def render(self, request):
                raise RuntimeError("gpu on fire")

        self.adapter_g.worker = Boom()
        self.adapter_a.dispatch_render("gpu-1", descriptor())
        pump(self.chain_a, self.chain_g)
        results = self._results()
        self.assertTrue(results, "a requester must never be left hanging")
        result = NRRRenderResult.from_dict(results[0].payload["result"])
        self.assertEqual(result.status, "not_supported")

    def test_node_ignores_its_own_envelopes(self):
        self.adapter_a._on_envelope(
            envelope("nrr_render_request", "agent-a",
                     {"request_id": "r1", "descriptor": descriptor()},
                     "nrr/render"), "loopback")
        self.assertEqual(self._results(), [])


# -- scene results feed the spatial index ----------------------------------

class TestSceneResultFusesIntoSpatialIndex(unittest.TestCase):

    def test_entities_land_in_spatial_index(self):
        bus = LoopbackBus("nrr-scene")
        chain_a = TransportChain("agent-a",
                                 [LoopbackTransport("agent-a", bus)])
        chain_g = TransportChain("gpu-1", [LoopbackTransport("gpu-1", bus)])
        index = SpatialIndex()
        spatial = SpatialMemoryNode("agent-a", chain_a, index)
        receiver = NRRMeshAdapter("agent-a", chain_a, spatial_sync=spatial)
        scene = NRRSceneResult(
            frame_id="f1", status="ok", source_device_id="cam-front",
            entities=[SceneEntity(
                id="robot-9", label="robot", confidence=0.9,
                position=Coordinate3D(x=3, y=4, z=0, frame="world_aligned"))])
        receiver._on_envelope(
            envelope("nrr_scene_result", "gpu-1",
                     {"type": "NRRSceneResult", "request_id": "r1",
                      "result": scene.to_dict()}, "nrr/scene_result"),
            "loopback")
        found = index.all_observations()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].entity_id, "robot-9")
        self.assertEqual(found[0].x, 3)

    def test_malformed_scene_result_is_ignored(self):
        chain = TransportChain("agent-a", [LoopbackTransport("agent-a",
                                    LoopbackBus("nrr-scene-bad"))])
        index = SpatialIndex()
        spatial = SpatialMemoryNode("agent-a", chain, index)
        adapter = NRRMeshAdapter("agent-a", chain, spatial_sync=spatial)
        adapter._on_envelope(
            envelope("nrr_scene_result", "gpu-1",
                     {"result": {"entities": "not-a-list"}}, "nrr/scene_result"),
            "loopback")
        self.assertEqual(index.all_observations(), [])


# -- descriptor contract ----------------------------------------------------

class TestDescriptorContract(unittest.TestCase):

    def test_round_trip(self):
        desc = NRRFrameDescriptor.from_dict(descriptor())
        self.assertEqual(desc.frame_id, "frame-1")
        self.assertEqual(desc.width, 320)
        self.assertEqual(NRRFrameDescriptor.from_dict(desc.to_dict()), desc)

    def test_bounds_enforced(self):
        for bad in ({"frame_id": "f" * 65, "width": 1, "height": 1},
                    {"frame_id": "f", "width": 99999, "height": 1},
                    {"frame_id": "f", "width": 1, "height": 1,
                     "quality_hint": "ultra"},
                    {"frame_id": "f", "width": 1, "height": 1,
                     "delta_time": -1.0}):
            with self.assertRaises(ValueError, msg=bad):
                NRRFrameDescriptor.from_dict(bad)

    def test_motion_request_requires_frame_id(self):
        # from_dict is lenient; validate() is the gate the adapter calls.
        with self.assertRaises(ValueError):
            NRRMotionRequest.from_dict({"frame_id": ""}).validate()
        with self.assertRaises(ValueError):
            NRRMotionRequest.from_dict(
                {"frame_id": "f1", "motion_threshold": 5.0}).validate()

    def test_result_construction_never_implies_pixels(self):
        # The result carries a handle + stats, never frame content.
        result = NRRRenderResult(frame_id="f1", status="ok")
        self.assertEqual(result.output_handle, "")
        self.assertEqual(result.width, 0)
        self.assertEqual(result.height, 0)
        self.assertEqual(set(result.to_dict()) & PIXEL_KEYS, set())
        self.assertNotIn("has_data", result.to_dict())


# -- runtime integration ----------------------------------------------------

class TestRuntimeNRRWiring(unittest.TestCase):
    """The adapter must be live on a connected runtime, not just standalone."""

    def _worker(self, seen=None):
        class Worker:
            backend = "gpu"
            available = True

            def render(self, request):
                if seen is not None:
                    seen.append(request)
                d = request["descriptor"]
                return {"type": "NRRResult",
                        "request_id": request["request_id"],
                        "result": NRRRenderResult(
                            frame_id=d["frame_id"], status="ok",
                            output_handle="local://" + d["frame_id"],
                            width=d["width"], height=d["height"]).to_dict()}
        return Worker()

    def _runtime(self, agent_id, worker=None, registry=None, **kw):
        from shugonet_runtime import ShugonetAgentRuntime
        kw.setdefault("nrr_worker", worker)
        return ShugonetAgentRuntime(agent_id, registry=registry, **kw)

    def test_worker_advertises_its_capability_in_the_manifest(self):
        agent = self._runtime("agent-a", worker=self._worker())
        caps = agent.manifest["compute_caps"]
        self.assertIn("nrr_render", caps["workloads"])
        self.assertEqual(caps["backend"], "gpu")
        status = agent.nrr_status()
        self.assertTrue(status["worker"])
        self.assertTrue(status["available"])

    def test_no_worker_means_no_capability_claim(self):
        agent = self._runtime("agent-a")
        self.assertNotIn("compute_caps", agent.manifest)
        self.assertFalse(agent.nrr_status()["worker"])

    def test_refuses_while_disconnected(self):
        agent = self._runtime("agent-a", worker=self._worker())
        self.assertIsNone(agent.nrr)
        for call in (lambda: agent.render_frame("gpu-1", descriptor()),
                     lambda: agent.query_scene("gpu-1", {"frame_id": "f"}),
                     lambda: agent.publish_motion_event({"event_id": "e"})):
            self.assertEqual(call()["status"], "refused")
            self.assertEqual(call()["reason"], "not connected")

    def test_capability_discovery_via_registry(self):
        registry = AgentRegistry()
        registry.pair("gpu-1", manifest={"compute_caps": {
            "workloads": ["nrr_render"]}})
        registry.pair("cam-1", manifest={"compute_caps": {
            "workloads": ["vision"]}})
        agent = self._runtime("agent-a", registry=registry)
        self.assertEqual([n["agent_id"] for n in agent.nrr_capable_nodes()],
                         ["gpu-1"])

    def test_end_to_end_render_over_a_live_host(self):
        """A connected pair must complete the full request/reply round trip."""
        from host import ShugonetHost
        seen = []
        host = ShugonetHost(agent_id="host-1")
        host.start()
        requester = responder = None
        try:
            host.registry.pair("agent-a", manifest={"realm": "phys"})
            host.registry.pair("gpu-1", manifest={
                "realm": "phys",
                "compute_caps": {"workloads": ["nrr_render"]}})
            requester = self._runtime("agent-a", registry=host.registry)
            responder = self._runtime("gpu-1", registry=host.registry,
                                      nrr_worker=self._worker(seen))
            for agent in (requester, responder):
                agent.host_tcp_host = "127.0.0.1"
                agent.host_tcp_port = host.tcp_port
                agent.host_agent_id = host.agent_id
                agent.host_relay_url = host.relay_url
                agent.connect_to_host()
            time.sleep(0.2)

            desc = descriptor()
            res = requester.render_frame("gpu-1", desc)
            self.assertEqual(res["status"], "success", res)
            for _ in range(60):
                requester.chain.poll(0.01)
                responder.chain.poll(0.01)
                time.sleep(0.01)

            # The peripheral worker ran, and only descriptors crossed the mesh.
            self.assertTrue(seen, "the peripheral worker was never invoked")
            self.assertEqual(seen[0]["descriptor"]["frame_id"],
                             desc["frame_id"])
            self.assertEqual(set(seen[0]) & PIXEL_KEYS, set())
        finally:
            for agent in (requester, responder):
                if agent is not None:
                    agent.stop()
            host.stop()


if __name__ == "__main__":
    unittest.main()
