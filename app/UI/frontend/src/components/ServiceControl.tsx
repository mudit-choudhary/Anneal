import { useEffect, useState } from "react";
import { api } from "../api";
import type { Service } from "../types";

function uptime(s?: number | null) {
  if (s == null) return "";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

/** Start, stop and restart every pipeline service without touching a terminal. */
export default function ServiceControl({ onChanged }: { onChanged?: () => void }) {
  const [services, setServices] = useState<Service[]>([]);
  const [busy, setBusy] = useState<string>("");
  const [msg, setMsg] = useState("");
  const [warn, setWarn] = useState(false);

  const reload = () =>
    api.services().then((r) => setServices(r.services)).catch(() => setServices([]));

  useEffect(() => {
    reload();
    const t = setInterval(reload, 4000);
    return () => clearInterval(t);
  }, []);

  async function act(name: string, action: "start" | "stop" | "restart", embed_device?: string) {
    setBusy(`${name}:${action}`);
    setMsg(name === "embedding" && action !== "stop"
      ? "Loading the embedding model — this takes up to a minute…" : "");
    setWarn(false);
    try {
      const r = await api.controlService(name, action, embed_device);
      setMsg(r.note ?? `${action} ${name} — ${r.running ? "running" : "stopped"}`);
      // asked for the GPU and got the CPU: that must not pass silently
      setWarn(Boolean(embed_device && r.device && r.device !== embed_device));
    } catch (e) {
      setMsg(`${(e as Error).message}`);
      setWarn(true);
    } finally {
      setBusy("");
      await reload();
      onChanged?.();
    }
  }

  async function preset(kind: "query" | "ingest") {
    setBusy(`preset:${kind}`);
    setMsg("Starting services…");
    setWarn(false);
    try {
      const r = await api.servicePreset(kind);
      const parts = [];
      if (r.started.length) parts.push(`started ${r.started.join(", ")}`);
      if (r.already_running.length) parts.push(`already up: ${r.already_running.join(", ")}`);
      setMsg([parts.join(" · ") || "nothing to do", r.note].filter(Boolean).join(" "));
    } catch (e) {
      setMsg(`${(e as Error).message}`);
    } finally {
      setBusy("");
      await reload();
      onChanged?.();
    }
  }

  return (
    <div className="card control">
      <div className="row wrap">
        <h2>Pipeline control</h2>
        <button className="small" disabled={busy !== ""} onClick={() => preset("query")}
                title="Registry + embedder on CPU — the GPU stays free for the answering model">
          Set up for questions
        </button>
        <button className="small" disabled={busy !== ""} onClick={() => preset("ingest")}
                title="Also parse and prune; embedder on the GPU">
          Set up for adding papers
        </button>
      </div>

      <table className="control-table">
        <tbody>
          {services.map((s) => (
            <tr key={s.name} className={s.running ? "" : "off"}>
              <td className="state">
                <span className={`pip ${s.running ? "on" : "off"}`} />
              </td>
              <td className="who">
                <div className="name">
                  {s.title}
                  {s.port && <span className="hint"> :{s.port}</span>}
                  {s.self && <span className="tag">this page</span>}
                </div>
                <div className="hint">{s.purpose}</div>
              </td>
              <td className="hint nowrap">
                {s.running ? `up ${uptime(s.uptime_seconds)} · pid ${s.pid}` : "stopped"}
              </td>
              <td className="acts">
                {s.running ? (
                  <>
                    <button className="small" disabled={s.self || busy !== ""}
                            title={s.self ? "Use the terminal: scripts/stop_services.sh" : "Restart"}
                            onClick={() => act(s.name, "restart")}>Restart</button>
                    <button className="small" disabled={s.self || busy !== ""}
                            title={s.self ? "The UI cannot stop itself" : "Stop"}
                            onClick={() => act(s.name, "stop")}>Stop</button>
                  </>
                ) : (
                  <button className="small" disabled={busy !== ""}
                          onClick={() => act(s.name, "start")}>Start</button>
                )}
                {s.name === "embedding" && s.running && (
                  <select className="mini" defaultValue="" disabled={busy !== ""}
                          title="The device is fixed when the embedder starts, so changing it restarts the service"
                          onChange={(e) => e.target.value && act("embedding", "restart", e.target.value)}>
                    <option value="" disabled>device…</option>
                    <option value="cpu">restart on CPU</option>
                    <option value="cuda">restart on GPU</option>
                  </select>
                )}
              </td>
            </tr>
          ))}
          {services.length === 0 && (
            <tr><td className="hint">Could not read service state.</td></tr>
          )}
        </tbody>
      </table>

      {msg && <p className={warn ? "warning" : "hint"}>{warn ? "⚠ " : ""}{msg}</p>}
      <p className="hint">
        Services keep running after this page is closed. The web UI is the one thing you cannot
        stop from here — use <code>scripts/stop_services.sh</code>.
      </p>
    </div>
  );
}
