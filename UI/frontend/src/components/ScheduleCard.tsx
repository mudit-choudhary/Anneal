import { useEffect, useState } from "react";
import { api } from "../api";
import type { ScheduleState, ScheduleTopic } from "../types";

function niceDate(s?: string | null) {
  if (!s) return null;
  const d = new Date(s);
  return isNaN(d.getTime()) ? s : d.toLocaleString(undefined,
    { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

/** A standing order: N papers a day for each topic on the list.
 *
 *  The topic list lives in settings, so the daily job reads it without the
 *  timer having to be reinstalled. Only the time of day is baked into the
 *  systemd unit. The switch reflects the timer's real state, not what we last
 *  asked for — a timer that failed to enable must not look enabled. */
export default function ScheduleCard() {
  const [s, setS] = useState<ScheduleState | null>(null);
  const [topics, setTopics] = useState<ScheduleTopic[]>([]);
  const [time, setTime] = useState("03:00");
  const [draft, setDraft] = useState("");
  const [draftN, setDraftN] = useState(10);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);

  const load = () =>
    api.schedule().then((r) => {
      setS(r);
      setTopics(r.topics);
      setTime(r.time);
      setDirty(false);
    }).catch((e) => setMsg(String(e.message)));

  useEffect(() => { load(); }, []);
  if (!s) return <div className="card"><h2>Daily downloads</h2><p className="hint">{msg || "Loading…"}</p></div>;

  const change = (next: ScheduleTopic[]) => { setTopics(next); setDirty(true); };

  function add() {
    const name = draft.trim();
    if (!name || topics.some((t) => t.topic.toLowerCase() === name.toLowerCase())) return;
    change([...topics, { topic: name, max_papers: draftN, enabled: true }]);
    setDraft("");
  }

  async function push(patch: { enabled?: boolean; save?: boolean }) {
    setBusy(true);
    setMsg("");
    try {
      const r = await api.saveSchedule({
        ...(patch.save ? { topics, time } : {}),
        ...(patch.enabled !== undefined ? { enabled: patch.enabled } : {}),
      });
      setS(r);
      setTopics(r.topics);
      setTime(r.time);
      setDirty(false);
      setMsg(r.note ?? "Saved.");
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const active = topics.filter((t) => t.enabled);
  const perDay = active.reduce((a, t) => a + t.max_papers, 0);
  const next = niceDate(s.next_run);
  const last = niceDate(s.last_run);

  return (
    <div className="card schedule">
      <div className="row wrap">
        <h2>Daily downloads</h2>
        <span className="spacer" />
        <label className="switch" title={s.enabled ? "Turn the daily job off" : "Turn the daily job on"}>
          <input type="checkbox" checked={s.enabled} disabled={busy}
                 onChange={(e) => push({ enabled: e.target.checked, save: true })} />
          <span>{s.enabled ? "On" : "Off"}</span>
        </label>
      </div>

      <p className={`verdict ${s.enabled ? "" : "off"}`}>
        {active.length === 0 ? (
          <>No topics yet. Add one below and the job has something to fetch.</>
        ) : s.enabled ? (
          <>Every day at <b>{time}</b>, up to <b>{perDay}</b> new papers
            across <b>{active.length}</b> topic{active.length === 1 ? "" : "s"}.</>
        ) : (
          <>Ready: <b>{perDay}</b> papers a day across <b>{active.length}</b> topic
            {active.length === 1 ? "" : "s"}, once you switch it on.</>
        )}
      </p>

      {topics.length > 0 && (
        <div className="topics">
          {topics.map((t, i) => (
            <div key={t.topic} className={`topic ${t.enabled ? "" : "muted"}`}>
              <input type="checkbox" checked={t.enabled}
                     onChange={(e) => change(topics.map((x, j) => j === i ? { ...x, enabled: e.target.checked } : x))} />
              <span className="name" title={t.topic}>{t.topic}</span>
              <span className="cap">
                <input type="number" min={1} max={200} value={t.max_papers}
                       onChange={(e) => change(topics.map((x, j) => j === i ? { ...x, max_papers: Number(e.target.value) } : x))} />
                <span className="hint">/day</span>
              </span>
              <button className="x" title="Remove this topic"
                      onClick={() => change(topics.filter((_, j) => j !== i))}>×</button>
            </div>
          ))}
        </div>
      )}

      <div className="search" style={{ marginTop: 10 }}>
        <input className="topic" value={draft} placeholder="Add a topic — e.g. Graph Neural Networks"
               onChange={(e) => setDraft(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && add()} />
        <input className="n" type="number" min={1} max={200} value={draftN}
               onChange={(e) => setDraftN(Number(e.target.value))} />
        <button className="small" disabled={!draft.trim()} onClick={add}>Add</button>
      </div>

      <div className="grid narrow" style={{ marginTop: 12 }}>
        <label>Run at</label>
        <span className="with-unit">
          <input type="time" value={time}
                 onChange={(e) => { setTime(e.target.value); setDirty(true); }} />
          <span className="unit">a missed run fires 5 min after the next boot</span>
        </span>
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <button className="send" disabled={busy || (!dirty && s.installed)} onClick={() => push({ save: true })}>
          {busy ? "Saving…" : dirty ? "Save changes" : "Saved"}
        </button>
        {(next || last) && (
          <span className="hint">
            {next && <>next {next}</>}{next && last && " · "}{last && <>last ran {last}</>}
          </span>
        )}
      </div>

      {s.enabled && !s.linger && (
        <p className="warning">
          ⚠ The job only runs while you are logged in. To keep it running after logout, once:
          <code>loginctl enable-linger $USER</code>
        </p>
      )}
      {msg && <p className="hint">{msg}</p>}
    </div>
  );
}
