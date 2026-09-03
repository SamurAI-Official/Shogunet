# Shogunet

> Networking integration for connecting multiple Shugocore agents together for
> collaboration between Shugocore agents in simulation and physical spaces,
> designed to be used over 5G, 4G, EDGE, LoRa, Wifi-Halow, Wifi, and Bluetooth
> networks.

[![PyPI](https://img.shields.io/pypi/v/shugonet)](https://pypi.org/project/shugonet/)
![Release](https://img.shields.io/badge/release-v0.4.0-blue)
![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Android%20%28Termux%2FChaquopy%29-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

Shogunet is the networking layer for [ShugoCore](https://github.com/SamurAI-Official/ShugoCore).
It lets a fleet of ShugoCore agents — running in Gazebo simulation, on servers,
on robots, or on Android handsets — discover each other, exchange tasks and
events, and consolidate a *codependent* memory mesh, over whatever networks the
mission has available, with deterministic fallback between them.

## Design principles

**One protocol, every network.** A single versioned envelope is spoken over all
transports, from 5G to LoRa. Two codecs exist for it: a JSON codec for
broadband links and a compact binary codec (16-byte header + TLV payload) for
constrained links such as LoRa (0.3–27 kbit/s, ~220-byte frames) and BLE.

**Memory invariants survive the network.** ShugoCore's memory tiers map to
network tiers: Tier 0/1 never leave the agent (N0). Tier 2 semantic facts sync
between paired agents (N1). Tier 3 is read-only everywhere: only *promotion
proposals* travel, and application stays operator-attributed (N2).

**Codependent memory.** Facts propagate with provenance
`(origin_agent_id, fact_id)`; salience merges additively; pruning propagates as
tombstones; and reinforcement is a *feedback loop* — when a peer's fact proves
useful, a tiny `reinforce` message flows back so useful memories survive decay
fleet-wide. Over LoRa, the loop runs on ~50-byte digest frames; bulk fact
content moves later over any broadband link (digest anti-entropy).

**Pairing is consent.** Only operator-allowlisted agent IDs pair, grants carry
TTLs, topics are namespaced (`/shugunet/{agent_id}/{topic}`), and every
ingress/egress field is sanitized and size-capped. Every network event lands in
a hash-chained audit log.

**Deterministic fallback.** Transports form an ordered, health-ranked chain
with per-transport circuit breakers. If the LAN drops, traffic falls to the
relay; if everything drops, a bounded store-and-forward WAL replays on
reconnect. Total transport exhaustion latches the ShugoCore governor through
its fallback triggers instead of failing silently.

## Network tiers

| Class | Networks | Protocol response |
|---|---|---|
| Broadband IP | 5G, 4G, WiFi | JSON codec, mesh query fan-out, bulk sync; LAN = direct TCP, cellular = relay hub |
| Midband IP | EDGE, WiFi-Halow | Compact codec, batching, latency-tolerant profiles |
| Constrained | LoRa | 16-byte header frames ≤ 220 B, P0/P1 priority only, digest anti-entropy, duty-cycle honored |
| Short-range | Bluetooth (RFCOMM/BLE) | Stream framing or segmented compact codec; pairing-based discovery |

Message priority classes: **P0** control (announce/heartbeat/ack/tombstone) →
all links · **P1** memory deltas (facts/reinforce/digests) → constrained-capable
· **P2** tasks and queries → IP links · **P3** bulk snapshots and audit
shipping → broadband only, deferred under low power / thermal pressure.

## Module map

| Module | Responsibility |
|---|---|
| `protocol.py` | Envelope schema, 16-byte binary header, JSON + compact TLV codecs, segmentation, validation |
| `transports.py` | `BaseTransport`, `LinkProfile` registry (7 networks + loopback), loopback bus |
| `link_simulator.py` | In-process impairment: bandwidth, latency, jitter, loss, MTU, duty cycle |
| `discovery.py` | Per-medium peer discovery (multicast beacons, hub rendezvous, pairing, duty-cycled LoRa) |
| `agent_registry.py` | Pairing = consent, allowlists, TTL grants, topic ACLs, sim/phys manifests |
| `tcp_transport.py` | Length-prefixed TCP transport with announce handshake (WiFi / WiFi-Halow profiles) |
| `relay_transport.py` / `relay_server.py` | Cellular path: HTTPS relay hub + long-poll client (5G/4G/EDGE, cross-NAT) |
| `lora_transport.py` | Serial SX126x/SX127x point-to-point transport (optional `pyserial`) |
| `bluetooth_transport.py` | RFCOMM (AF_BLUETOOTH) and BLE (optional `bleak`) transports |
| `transport_fallback.py` | Link-aware chain: eligibility, EWMA health ranking, breakers, QoS |
| `store_forward.py` | Bounded JSONL WAL outbox/inbox with replay + dedup |
| `memory_sync.py` / `mesh_query.py` | Codependent memory mesh: fact replication, conflict resolution, fan-out queries |
| `audit.py`, `fallbacks.py`, `policy.py`, `telemetry.py` | ShugoCore-aligned safety surface |
| `shugocore_bridge.py` | Adapter hosting a Shogunet node beside a ShugoCore `DecisionEngine` |

## Hosting a fleet

Shogunet runs as a single server process that owns the fleet's trust plane so
every ShugoCore agent only needs a ~20-line client runtime to join.

### Start a host

```python
from host import ShugonetHost

host = ShugonetHost(agent_id="fleet-1", tcp_port=9000, relay_port=9001)
host.start()

# Grant consent to a joining agent (pairing = consent)
host.pair("agent-001", manifest={"realm": "phys", "role": "perception"})

print(host.status())   # roster, alive count, breaker health, mesh counts
```

Or from the CLI:

```bash
python3 host.py --tcp-port 9000 --relay-port 9001
```

### Join from a ShugoCore process

```python
from agent_runtime import ShugonetAgentRuntime

runtime = ShugonetAgentRuntime(
    agent_id="agent-001",
    host_tcp_host="127.0.0.1",
    host_tcp_port=9000,
    host_relay_url="http://127.0.0.1:9001",
    on_message=lambda sender, msg: print(f"from {sender}: {msg}"))
runtime.connect_to_host()

# Send a task to another agent
runtime.send("agent-002", "/shugunet/agent-001/task",
             {"action": "scan", "zone": "north"})

# Search the fleet's memory
results = runtime.query("obstacle in zone north")

# Sync memory with the fleet
runtime.sync()

# Leave the fleet
runtime.stop()
```

### Routing model

Agents connect **to** the host (hub-and-spoke). The host forwards addressed
mail to its recipient and fans broadcasts out as addressed copies to every
other paired agent. Host-addressed mail — heartbeats, mesh queries, memory
broadcasts — is processed locally by the host's own chain handlers.

**Pairing is enforced twice** — at TCP admission (handshake hook) and again on
every host-processed or forwarded envelope (the relay path has no registry gate
of its own, so the host re-checks the sender there).

### Failure semantics

- **Agent crash/disconnect** → host governor latches `pause` (ShugoCore
  deterministic latch contract).
- **Message durability** → per-agent `OutboxStore` + `at_least_once` QoS + hub
  mailbox TTL bounds.
- **Cross-talk** → chain and mesh recipient guards (hardened in the
  concurrency suite) hold at host scale.

### Fleet dashboard

Every `ShugunetHost` can expose an operator console — a compiled single-page
app (TypeScript + SolidJS, built with Vite) served directly by the host over a
loopback HTTP port. No JavaScript toolchain is needed at install time: the
built assets ship inside the wheel (`shugonet_web/static`).

```python
host = ShugonetHost(agent_id="fleet-1", tcp_port=9000, relay_port=9001,
                    dashboard_port=9002)
host.start()
# open http://127.0.0.1:9002 in a browser
```

The console streams a live event feed (Server-Sent Events), shows the roster,
transport-chain health, memory-mesh counters, and audit-chain integrity, and
exposes pair / unpair / resume / broadcast controls. State-changing POSTs are
audited and can be gated by a token (`dashboard_token=...`).

The SPA source lives in `dashboard/`; rebuild it with:

```bash
cd dashboard && npm ci && npm run build
```

## Module map (continued)

| Module | Responsibility |
|---|---|
| `host.py` | `ShugunetHost`: admit paired agents, route traffic, seed the mesh |
| `agent_runtime.py` | `ShugonetAgentRuntime`: client half a ShugoCore process instantiates |
| `dashboard.py` | `DashboardServer`: stdlib HTTP operator plane (REST + SSE + SPA) |
| `pg_store.py` | `PgFactStore`: optional PostgreSQL mesh backend (ShugoCore `PgSemanticMemory` parity) |

## Testing

```bash
python3 -m unittest discover -s tests -v
```

The suite is dependency-free and runs everywhere: transports are exercised
through the loopback bus and the in-process link simulator, which models every
network's bandwidth, latency, loss, MTU and duty-cycle constraints. The
integration suite spins up a real `ShugonetHost` with multiple threaded
`ShugonetAgentRuntime` clients, validates cross-talk isolation, memory
convergence, peer-lost latching, and unpaired-agent refusal.

## Changelog

### 0.4.0

- **Fleet dashboard**: stdlib HTTP operator plane (`dashboard.py`) with a
  compiled TypeScript + SolidJS SPA (`dashboard/` → `shugonet_web/static`).
  Serves JSON REST, a live SSE event stream, and the console; ships inside the
  wheel so no JS toolchain is needed at install time.
- **ShugoCore memory compatibility**: fact schema aligned with
  `SemanticMemory` / `PgSemanticMemory` columns (`kind`, `metadata`,
  `created_at`); `pg_store.PgFactStore` mirrors mesh facts into PostgreSQL
  (optional `postgres` extra).
- **Version handshake**: `shugonet_version` + `protocol_version` exchanged at
  TCP admission; mismatches emit a `version_mismatch` audit event and refuse
  the join.
- **Bridge sync test**: `tests/test_bridge_sync.py` keeps Shogunet's
  `shugocore_adapter` and ShugoCore's vendored `shugonet_bridge` from drifting.
- Version bumped 0.1.0 → 0.4.0.
