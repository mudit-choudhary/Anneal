import { useEffect, useState } from "react";
import { api } from "../api";
import { bytes, eta } from "../format";
import CoverageCard from "./CoverageCard";
import GetPapers from "./GetPapers";
import PruneCard from "./PruneCard";
import ScheduleCard from "./ScheduleCard";
import ServiceControl from "./ServiceControl";
import type { Ingestion as IngestionT } from "../types";

const STAGES = [
  { key: "downloaded", label: "downloaded", hint: "waiting to be parsed" },
  { key: "parsed", label: "parsed", hint: "layout detected" },
  { key: "processed", label: "processed", hint: "text assembled" },
  { key: "embedded", label: "embedded", hint: "searchable" },
  { key: "error", label: "error", hint: "needs attention" },
];

export default function Ingestion() {
  const [data, setData] = useState<IngestionT | null>(null);

  const reload = () => api.ingestion().then(setData).catch(() => setData(null));
  useEffect(() => {
    reload();
    const t = setInterval(reload, 5000);
    return () => clearInterval(t);
  }, []);

  const reg = data?.registry;
  const total = reg?.total ?? 0;
  const done = reg?.counts.embedded ?? 0;
  const remaining = data?.remaining ?? 0;
  const pct = total ? Math.round((100 * done) / total) : 0;

  return (
    <div className="page ingestion">
      {/* one clear headline, then the detail */}
      <div className="card hero">
        {!reg ? (
          <p className="hint">
            Registry not reachable — start the pipeline with
            <code> anneal --with-ingest</code>.
          </p>
        ) : (
          <>
            <div className="hero-line">
              <div>
                <span className="big">{done}</span>
                <span className="of"> / {total} papers searchable</span>
              </div>
              <div className="hero-right">
                {remaining > 0 ? (
                  <>
                    <b>{remaining}</b> in the pipeline
                    {data?.pending_bytes ? <> · {bytes(data.pending_bytes)}</> : null}
                    {data?.eta_seconds != null && <> · ETA <b>{eta(data.eta_seconds)}</b></>}
                  </>
                ) : (
                  <span className="ok">everything is processed</span>
                )}
              </div>
            </div>
            <div className="progress"><div style={{ width: `${pct}%` }} /></div>

            {data?.eta_note && <p className="warning">⚠ {data.eta_note}</p>}
            {data?.throughput_per_hour != null && (
              <p className="hint">Measured throughput: {data.throughput_per_hour} papers/hour.</p>
            )}

            <div className="stage-row">
              {STAGES.map((s) => {
                const n = reg.counts[s.key] ?? 0;
                return (
                  <div key={s.key} className={`stage ${s.key} ${n ? "" : "empty"}`} title={s.hint}>
                    <div className="n">{n}</div>
                    <div className="label">{s.label}</div>
                  </div>
                );
              })}
            </div>

            {reg.errors.length > 0 && (
              <details className="errors" open>
                <summary>{reg.errors.length} paper(s) failed</summary>
                {reg.errors.slice(0, 20).map((e) => (
                  <div key={e.filename}><code>{e.filename}</code> ×{e.error_count}: {e.last_error}</div>
                ))}
                <p className="hint">Retry with <code>python scripts/register_pdfs.py --retry-errors</code></p>
              </details>
            )}
          </>
        )}
      </div>

      <ServiceControl onChanged={reload} />

      <CoverageCard onRepaired={reload} />

      <GetPapers onStarted={reload} />

      <div className="cards">
        <ScheduleCard />
        <PruneCard />
      </div>

      <div className="card">
        <div className="row">
          <h2>Files on disk</h2>
          <span className="hint">
            not the same as the stages above — those say where each paper *is*, these are the
            files still on disk
          </span>
        </div>
        <div className="disk-row">
          {Object.entries(data?.files ?? {}).map(([k, v]) => (
            <div key={k} className="disk">
              <div className="n">{v.papers}</div>
              <div className="label">{k.replace("_", " ")}</div>
              <div className="hint">
                {v.files} file{v.files === 1 ? "" : "s"}
                {v.files !== v.papers && v.papers > 0 && ` · ${(v.files / v.papers).toFixed(0)} per paper`}
                {v.bytes ? ` · ${bytes(v.bytes)}` : ""}
              </div>
            </div>
          ))}
        </div>
        <p className="hint">
          <b>processed</b> holds two files per paper — the readable <code>.txt</code> and the
          <code>.json</code> blocks the embedder reads — so its file count is double its paper
          count. Parsed and processed files stay until the pruner removes them, which is why
          they can still be here when every paper is already embedded.
        </p>
      </div>
    </div>
  );
}
