import { useEffect, useMemo, useRef, useState } from "react";
import { streamLogs } from "../api";

const SERVICES = ["parse", "embedding", "download", "registry", "prune", "ui", "daily-ingest"];
const LEVELS = ["all", "info", "warning", "error"] as const;
type Level = (typeof LEVELS)[number];

function levelOf(line: string): Level {
  const l = line.toLowerCase();
  if (l.includes("error") || l.includes("traceback") || l.includes("critical")) return "error";
  if (l.includes("warn")) return "warning";
  return "info";
}

/** Live tail of run/logs/<service>.log, with a filter and search. */
export default function Logs() {
  const [service, setService] = useState("parse");
  const [lines, setLines] = useState<string[]>([]);
  const [level, setLevel] = useState<Level>("all");
  const [q, setQ] = useState("");
  const [follow, setFollow] = useState(true);
  const [connected, setConnected] = useState(true);
  const boxRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    setLines([]);
    setConnected(true);
    const stop = streamLogs(service, 400, (line) => setLines((l) => [...l.slice(-4999), line]),
                            () => setConnected(false));
    return stop;
  }, [service]);

  const shown = useMemo(() => {
    const needle = q.toLowerCase();
    return lines.filter((l) => (level === "all" || levelOf(l) === level) &&
                               (!needle || l.toLowerCase().includes(needle)));
  }, [lines, level, q]);

  useEffect(() => {
    if (follow) boxRef.current?.scrollTo({ top: boxRef.current.scrollHeight });
  }, [shown, follow]);

  return (
    <div className="page logs-page">
      <div className="card">
        <div className="row wrap">
          <h2>Logs</h2>
          <select value={service} onChange={(e) => setService(e.target.value)}>
            {SERVICES.map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={level} onChange={(e) => setLevel(e.target.value as Level)}>
            {LEVELS.map((l) => <option key={l} value={l}>{l === "all" ? "all levels" : l}</option>)}
          </select>
          <input type="search" placeholder="Filter…" value={q} onChange={(e) => setQ(e.target.value)}
                 style={{ flex: "1 1 200px" }} />
          <label className="inline">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow
          </label>
          <button className="small" onClick={() => setLines([])}>clear</button>
          <span className="hint">
            {shown.length}{shown.length !== lines.length && ` of ${lines.length}`} line(s)
            {!connected && " · disconnected"}
          </span>
        </div>
        <pre ref={boxRef} className="logbox">
          {shown.length
            ? shown.map((l, i) => <div key={i} className={`log ${levelOf(l)}`}>{l}</div>)
            : <span className="hint">
                {lines.length ? "Nothing matches the filter." : `No output yet from ${service}.`}
              </span>}
        </pre>
        <p className="hint">
          Tailing <code>run/logs/{service}.log</code>. A service that was never started has no log file.
        </p>
      </div>
    </div>
  );
}
