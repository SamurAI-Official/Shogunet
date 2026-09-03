import { createSignal, For, Show } from "solid-js";
import { api, type AuditRecord, type HostStatus } from "./api";

const MODE_CLASS: Record<string, string> = {
  normal: "ok",
  paused: "warn",
  safe_state: "warn",
  halted: "crit",
  unknown: "dim",
};

export function FleetHeader(props: { status: () => HostStatus | null }) {
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal("");
  const s = () => props.status();

  const resume = async () => {
    const who = window.prompt("Resume attribution (your name / ticket):");
    if (!who) return;
    setBusy(true);
    setError("");
    try {
      await api.resume(who);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  };

  const broadcast = async () => {
    const text = window.prompt("System broadcast to the fleet:");
    if (!text) return;
    setBusy(true);
    setError("");
    try {
      await api.broadcast({ notice: text });
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  };

  return (
    <header class="header">
      <div class="brand">
        <span class="brand-name">Shugonet</span>
        <span class="brand-sub">fleet console</span>
        <Show when={s()}>
          <span class="pill dim">v{s()!.shugonet_version}</span>
          <span class="pill dim">wire v{s()!.protocol_version}</span>
        </Show>
      </div>
      <div class="header-mid">
        <Show when={s()} fallback={<span class="pill dim">connecting…</span>}>
          <span class={`pill ${MODE_CLASS[s()!.mode] ?? "dim"}`}>
            {s()!.mode}
          </span>
          <Show when={s()!.mode_reason}>
            <span class="mode-reason" title={s()!.mode_reason}>
              {s()!.mode_reason}
            </span>
          </Show>
          <span class="stat">
            alive <b>{s()!.alive_count}</b>/{s()!.paired_count}
          </span>
          <span class="stat">
            facts <b>{s()!.store_count}</b>
          </span>
          <span class="stat">
            audit <b>{s()!.audit_len}</b>
          </span>
        </Show>
      </div>
      <div class="header-actions">
        <span class="error">{error()}</span>
        <button class="btn" disabled={busy()} onClick={broadcast}>
          Broadcast
        </button>
        <Show when={s() && s()!.mode !== "normal"}>
          <button class="btn primary" disabled={busy()} onClick={resume}>
            Resume…
          </button>
        </Show>
      </div>
    </header>
  );
}

export function Roster(props: {
  status: () => HostStatus | null;
  onChanged: () => void;
}) {
  const [busy, setBusy] = createSignal("");
  const [error, setError] = createSignal("");
  const s = () => props.status();

  const pair = async () => {
    const agent_id = window.prompt("Agent ID to pair:");
    if (!agent_id) return;
    setBusy(agent_id);
    setError("");
    try {
      await api.pair(agent_id, "dashboard-operator");
      props.onChanged();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy("");
    }
  };

  const unpair = async (agent_id: string) => {
    if (!window.confirm(`Unpair ${agent_id}?`)) return;
    setBusy(agent_id);
    setError("");
    try {
      await api.unpair(agent_id);
      props.onChanged();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy("");
    }
  };

  return (
    <section class="panel">
      <div class="panel-head">
        <h2>Agents</h2>
        <button class="btn small" onClick={pair}>Pair…</button>
      </div>
      <span class="error">{error()}</span>
      <Show when={s() && s()!.roster.length > 0}
            fallback={<p class="empty">No agents paired.</p>}>
        <div class="agent-grid">
          <For each={s()!.roster}>
            {(agent) => (
              <div class={`agent-card ${agent.alive ? "" : "stale"}`}>
                <div class="agent-top">
                  <span class={`dot ${agent.alive ? "ok" : "crit"}`} />
                  <span class="agent-id" title={agent.agent_id}>
                    {agent.agent_id}
                  </span>
                  <span class="pill dim">{agent.realm}</span>
                </div>
                <dl>
                  <dt>version</dt><dd>{agent.client_version}</dd>
                  <dt>paired by</dt><dd>{agent.paired_by}</dd>
                  <dt>expires</dt>
                  <dd>{new Date(agent.expires_at * 1000).toLocaleTimeString()}</dd>
                </dl>
                <button class="btn small danger"
                        disabled={busy() === agent.agent_id}
                        onClick={() => unpair(agent.agent_id)}>
                  Unpair
                </button>
              </div>
            )}
          </For>
        </div>
      </Show>
    </section>
  );
}

export function ChainPanel(props: { status: () => HostStatus | null }) {
  const s = () => props.status();
  const chain = () => (s()?.chain ?? {}) as Record<string, number>;
  const rows = () => (s()?.chain_health ?? []) as {
    transport: string;
    profile: string;
    breaker_open: boolean;
    consecutive_failures: number;
    latency_ewma_s: number | null;
    accepts: number[];
  }[];
  const num = (key: string) => chain()[key] ?? 0;

  return (
    <section class="panel">
      <div class="panel-head"><h2>Transport chain</h2></div>
      <div class="counters">
        <span>sent <b>{num("sent")}</b></span>
        <span>received <b>{num("received")}</b></span>
        <span>failed <b>{num("send_failed")}</b></span>
      </div>
      <table class="table">
        <thead>
          <tr>
            <th>transport</th><th>profile</th><th>breaker</th>
            <th>latency</th><th>classes</th>
          </tr>
        </thead>
        <tbody>
          <For each={rows()}>
            {(row) => (
              <tr>
                <td>{row.transport}</td>
                <td class="dim">{row.profile}</td>
                <td>
                  <span class={`dot ${row.breaker_open ? "crit" : "ok"}`} />
                  {" "}{row.breaker_open ? "open" : "closed"}
                  <Show when={row.consecutive_failures > 0}>
                    <span class="dim"> ×{row.consecutive_failures}</span>
                  </Show>
                </td>
                <td>{row.latency_ewma_s == null
                  ? "—"
                  : `${(row.latency_ewma_s * 1000).toFixed(0)} ms`}</td>
                <td class="dim">{row.accepts.join(", ")}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
      <Show when={rows().length === 0}>
        <p class="empty">Chain not started.</p>
      </Show>
    </section>
  );
}

export function MeshPanel(props: { status: () => HostStatus | null }) {
  const s = () => props.status();
  const mesh = () => (s()?.mesh ?? {}) as Record<string, number>;
  const rows = () => Object.entries(mesh())
    .filter(([, v]) => typeof v === "number")
    .sort((a, b) => a[0].localeCompare(b[0]));

  return (
    <section class="panel">
      <div class="panel-head">
        <h2>Memory mesh</h2>
        <span class="pill dim">{s()?.store_count ?? 0} facts</span>
      </div>
      <div class="counters">
        <For each={rows()}>
          {([key, value]) => (
            <span>{key.replace(/_/g, " ")} <b>{value}</b></span>
          )}
        </For>
      </div>
    </section>
  );
}

export function AuditPanel(props: { status: () => HostStatus | null }) {
  const s = () => props.status();
  return (
    <section class="panel">
      <div class="panel-head"><h2>Audit chain</h2></div>
      <div class="counters">
        <span>length <b>{s()?.audit_len ?? 0}</b></span>
        <span class="mono dim tail" title={s()?.audit_tail ?? ""}>
          tail {(s()?.audit_tail ?? "—").slice(0, 12)}…
        </span>
      </div>
    </section>
  );
}

const NICE_EVENTS: Record<string, string> = {
  agent_paired: "ok",
  agent_unpaired: "warn",
  agent_pairing_expired: "warn",
  host_mode_changed: "warn",
  version_mismatch: "crit",
  tcp_admission_refused: "crit",
  unpaired_sender_dropped: "crit",
  memory_conflict_resolved: "warn",
  network_fallback_trigger: "warn",
};

export function EventFeed(props: { events: () => AuditRecord[] }) {
  const fmt = (record: AuditRecord) => {
    const bits = Object.entries(record.payload ?? {})
      .slice(0, 3)
      .map(([key, value]) => `${key}=${String(value).slice(0, 48)}`);
    return bits.join("  ");
  };

  return (
    <section class="panel grow">
      <div class="panel-head"><h2>Live events</h2></div>
      <div class="feed">
        <For each={props.events()}>
          {(record) => (
            <div class={`feed-row ${NICE_EVENTS[record.event] ?? ""}`}>
              <span class="feed-seq">#{record.seq}</span>
              <span class="feed-event">{record.event}</span>
              <span class="feed-detail">{fmt(record)}</span>
            </div>
          )}
        </For>
        <Show when={props.events().length === 0}>
          <p class="empty">Waiting for events…</p>
        </Show>
      </div>
    </section>
  );
}