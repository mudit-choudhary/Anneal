import { useEffect, useRef, useState } from "react";
import { api, streamLogs } from "../api";
import type { Ingestion as IngestionT } from "../types";

const STATUSES = ["downloaded", "parsed", "processed", "embedded", "error"];
const SERVICES = ["registry", "parse", "embedding", "prune", "ui", "download", "daily-ingest"];

function eta(seconds: number | null) {
  if (seconds == null) return "—";
  if (seconds < 90) return `~${seconds}s`;
  if (seconds < 5400) return `~${Math.round(seconds / 60)} min`;
  return `~${(seconds / 3600).toFixed(1)} h`;
}

export default function Ingestion() {
  const [data, setData] = useState<IngestionT | null>(null);
  const [service, setService] = useState("parse");
  const [lines, setLines] = useState<string[]>([]);
  const [paused, setPaused] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const load = () => api.ingestion().then(setData).catch(() => setData(null));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    setLines([]);
    const stop = streamLogs(service, 200, (line) => setLines((l) => [...l.slice(-1999), line]));
    return stop;
  }, [service]);

  useEffect(() => {
    if (!paused) logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [lines, paused]);

  const reg = data?.registry;
  const total = reg?.total ?? 0;
  const done = (reg?.counts.embedded ?? 0) + (reg?.counts.error ?? 0);

  return (
    <div className="page ingestion">
      <div className="cards">
        <div className="card">
          <h2>Progress</h2>
          {!reg ? (
            <p className="hint">Registry not reachable — start the pipeline (scripts/start_query.sh --with-ingest or fresh_start.sh).</p>
          ) : (
            <>
              <div className="progress"><div style={{ width: total ? `${(100 * done) / total}%` : "0%" }} /></div>
              <p>
                <b>{done}</b> of <b>{total}</b> papers finished · remaining {data?.remaining ?? 0} · ETA <b>{eta(data?.eta_seconds ?? null)}</b>
              </p>
              <table className="counts">
                <tbody>
                  {STATUSES.map((s) => (
                    <tr key={s} className={s}>
                      <td>{s}</td>
                      <td>{reg.counts[s] ?? 0}</td>
                      <td><div className="bar"><div style={{ width: total ? `${(100 * (reg.counts[s] ?? 0)) / total}%` : 0 }} /></div></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="hint">
                avg per paper — parse {fmt(reg.stage_seconds.parse)} · assemble {fmt(reg.stage_seconds.process)} · embed {fmt(reg.stage_seconds.embed)}
              </p>
              {reg.errors.length > 0 && (
                <div className="errors">
                  <h3>Errors ({reg.errors.length})</h3>
                  {reg.errors.slice(0, 20).map((e) => (
                    <div key={e.filename}><code>{e.filename}</code> ×{e.error_count}: {e.last_error}</div>
                  ))}
                  <p className="hint">Retry with: python scripts/register_pdfs.py --retry-errors</p>
                </div>
              )}
            </>
          )}
        </div>

        <div className="card">
          <h2>Services</h2>
          <table className="services">
            <tbody>
              {Object.entries(data?.services ?? {}).map(([name, s]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className={s.running ? "ok" : "bad"}>{s.running ? "running" : "stopped"}</td>
                  <td className="hint">pid {s.pid}</td>
                </tr>
              ))}
              {Object.keys(data?.services ?? {}).length === 0 && (
                <tr><td className="hint">none started via the ops scripts</td></tr>
              )}
            </tbody>
          </table>
          <h2>Files on disk</h2>
          <table className="services">
            <tbody>
              {Object.entries(data?.files ?? {}).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card logs">
        <div className="row">
          <h2>Logs</h2>
          <select value={service} onChange={(e) => setService(e.target.value)}>
            {SERVICES.map((s) => <option key={s}>{s}</option>)}
          </select>
          <label className="inline"><input type="checkbox" checked={paused} onChange={(e) => setPaused(e.target.checked)} /> pause autoscroll</label>
          <button className="small" onClick={() => setLines([])}>clear</button>
        </div>
        <pre ref={logRef}>{lines.join("\n") || "(no output yet)"}</pre>
      </div>
    </div>
  );
}

function fmt(v: number | null | undefined) {
  if (v == null) return "—";
  return v < 60 ? `${v.toFixed(0)}s` : `${(v / 60).toFixed(1)} min`;
}
