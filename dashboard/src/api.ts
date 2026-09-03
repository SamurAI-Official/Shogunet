// Typed REST + SSE client for the Shugonet dashboard API.

export interface AgentEntry {
  agent_id: string;
  realm: string;
  alive: boolean;
  client_version: string;
  paired_by: string;
  paired_at: number;
  expires_at: number;
  manifest: Record<string, unknown>;
}

export interface HostStatus {
  agent_id: string;
  shugonet_version: string;
  protocol_version: number;
  mode: string;
  mode_reason: string;
  tcp_port: number;
  relay_url: string | null;
  dashboard_url: string | null;
  roster: AgentEntry[];
  paired_count: number;
  alive_count: number;
  fallback: { mode: string; violations: Record<string, number> };
  chain: Record<string, unknown>;
  chain_health: {
    transport: string;
    profile: string;
    breaker_open: boolean;
    consecutive_failures: number;
    latency_ewma_s: number | null;
    accepts: number[];
  }[];
  mesh: Record<string, number>;
  store_count: number;
  audit_len: number;
  audit_tail: string;
}

export interface AuditRecord {
  seq: number;
  ts: string;
  event: string;
  payload: Record<string, unknown>;
}

async function request<T>(path: string, body?: unknown,
                          method: string = "GET"): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: body !== undefined
      ? { "Content-Type": "application/json" }
      : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const data = (await response.json().catch(() => ({}))) as T & {
    reason?: string;
  };
  if (!response.ok) {
    throw new Error((data as { reason?: string }).reason
      ?? `HTTP ${response.status}`);
  }
  return data;
}

export const api = {
  status: () => request<HostStatus>("/api/status"),
  audit: (limit = 200) =>
    request<{ records: AuditRecord[]; problems: unknown[] }>(
      `/api/audit?limit=${limit}`),
  pair: (agent_id: string, paired_by: string, manifest?: object) =>
    request<{ ok: boolean }>("/api/pair",
      { agent_id, paired_by, manifest: manifest ?? {} }, "POST"),
  unpair: (agent_id: string) =>
    request<{ ok: boolean }>("/api/unpair", { agent_id }, "POST"),
  resume: (resumed_by: string) =>
    request<{ ok: boolean }>("/api/resume", { resumed_by }, "POST"),
  broadcast: (payload: Record<string, unknown>) =>
    request<{ ok: boolean; reached: number }>("/api/broadcast",
      { payload }, "POST"),
};

/** Live status + audit events over SSE, with a polling safety net. */
export function openEventStream(
  onAudit: (record: AuditRecord) => void,
  onStatus: (status: HostStatus) => void,
): () => void {
  let source: EventSource | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;
  let closed = false;

  const startPolling = () => {
    if (pollTimer !== null) return;
    pollTimer = setInterval(() => {
      api.status().then(onStatus).catch(() => undefined);
    }, 5000);
  };
  const stopPolling = () => {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  };

  const connect = () => {
    if (closed) return;
    source = new EventSource("/api/events");
    source.addEventListener("audit", (ev) => {
      try {
        onAudit(JSON.parse((ev as MessageEvent).data));
      } catch { /* malformed frame is dropped */ }
    });
    source.addEventListener("status", (ev) => {
      attempt = 0;
      stopPolling();
      try {
        onStatus(JSON.parse((ev as MessageEvent).data));
      } catch { /* ignored */ }
    });
    source.onopen = () => { attempt = 0; };
    source.onerror = () => {
      source?.close();
      source = null;
      startPolling();          // safety net while the stream is down
      const backoff = Math.min(15000, 500 * 2 ** Math.min(attempt, 4));
      attempt += 1;
      retryTimer = setTimeout(connect, backoff);
    };
  };

  // Seed with an immediate snapshot, then ride the stream.
  api.status().then(onStatus).catch(() => undefined);
  connect();

  return () => {
    closed = true;
    source?.close();
    if (retryTimer !== null) clearTimeout(retryTimer);
    stopPolling();
  };
}