import { createSignal, onCleanup } from "solid-js";
import { render } from "solid-js/web";
import { api, openEventStream, type AuditRecord, type HostStatus } from "./api";
import { AuditPanel, ChainPanel, EventFeed, FleetHeader, MeshPanel,
         Roster } from "./panels";
import "./styles.css";

const MAX_FEED = 200;

function App() {
  const [status, setStatus] = createSignal<HostStatus | null>(null);
  const [events, setEvents] = createSignal<AuditRecord[]>([]);

  const refresh = () => api.status().then(setStatus).catch(() => undefined);

  onCleanup(openEventStream(
    (record) => {
      setEvents((previous) => {
        const next = [record, ...previous];       // newest first
        return next.length > MAX_FEED ? next.slice(0, MAX_FEED) : next;
      });
      refresh();                                   // counters ride along
    },
    (snapshot) => setStatus(snapshot),
  ));

  return (
    <div class="layout">
      <FleetHeader status={status} />
      <main class="columns">
        <div class="col">
          <Roster status={status} onChanged={refresh} />
          <ChainPanel status={status} />
        </div>
        <div class="col">
          <MeshPanel status={status} />
          <AuditPanel status={status} />
          <EventFeed events={events} />
        </div>
      </main>
    </div>
  );
}

const root = document.getElementById("root");
if (root) {
  render(() => <App />, root);
}
