import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { GpuInfo } from "../types";

const COLORS = ["#6ea8fe", "#3fb96b", "#d9a13b", "#a87bd6", "#4fa3a8"];

function SplitBar({ gpu }: { gpu: number }) {
  return (
    <div className="split" title={`${gpu}% on GPU, ${100 - gpu}% on CPU`}>
      <div className="split-gpu" style={{ width: `${gpu}%` }} />
      <span>{gpu === 100 ? "GPU" : gpu === 0 ? "CPU" : `${gpu}% GPU`}</span>
    </div>
  );
}

export default function Gpu() {
  const [g, setG] = useState<GpuInfo | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const load = () =>
      api.gpu().then((r) => {
        setG(r);
        if (r.device) setHistory((h) => [...h.slice(-119), r.device!.used_mb / r.device!.total_mb]);
      }).catch(() => setG(null));
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, []);

  // small sparkline of VRAM use over the last few minutes
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const w = (c.width = c.clientWidth * 2), h = (c.height = 96);
    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;
    const style = getComputedStyle(document.documentElement);
    ctx.strokeStyle = style.getPropertyValue("--accent").trim() || "#6ea8fe";
    ctx.lineWidth = 3;
    ctx.beginPath();
    history.forEach((v, i) => {
      const x = (i / (history.length - 1)) * w;
      const y = h - v * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }, [history]);

  if (!g) return <div className="page"><p className="hint">Reading GPU state…</p></div>;
  if (!g.available) {
    return (
      <div className="page">
        <div className="card">
          <h2>GPU</h2>
          <p className="hint">
            No NVIDIA GPU visible (<code>nvidia-smi</code> is missing or reported nothing).
            Everything still runs on the CPU, just more slowly.
          </p>
        </div>
      </div>
    );
  }

  const d = g.device!;
  const usedPct = Math.round((100 * d.used_mb) / d.total_mb);
  const tight = d.free_mb < 300;

  return (
    <div className="page gpu-page">
      <div className="card hero">
        <div className="hero-line">
          <div>
            <span className="big">{(d.used_mb / 1024).toFixed(1)}</span>
            <span className="of"> / {(d.total_mb / 1024).toFixed(1)} GB VRAM used</span>
          </div>
          <div className="hero-right">
            {d.name} · {d.utilisation_percent}% busy · {d.temperature_c}°C ·{" "}
            <b className={tight ? "bad" : "ok"}>{d.free_mb} MB free</b>
          </div>
        </div>

        {/* one bar, segmented by process */}
        <div className="vram-bar">
          {g.processes.map((p, i) => (
            <div key={p.pid} className="seg"
                 style={{ width: `${(100 * p.used_mb) / d.total_mb}%`, background: COLORS[i % COLORS.length] }}
                 title={`${p.owner} — ${p.used_mb} MB`} />
          ))}
        </div>
        <div className="legend">
          {g.processes.map((p, i) => (
            <span key={p.pid}>
              <i style={{ background: COLORS[i % COLORS.length] }} />
              {p.owner} <span className="hint">{p.used_mb} MB · pid {p.pid}</span>
            </span>
          ))}
          {g.processes.length === 0 && <span className="hint">No process is using the GPU.</span>}
          <span><i style={{ background: "var(--panel2)" }} />free <span className="hint">{d.free_mb} MB</span></span>
        </div>

        {tight && (
          <p className="warning">
            ⚠ Almost no VRAM left ({usedPct}% used). A model starting now will fall back to the
            CPU. Free it by stopping the answering model (<code>ollama stop</code>) or the
            embedder from the Ingestion tab.
          </p>
        )}
        <canvas ref={canvas} className="spark" style={{ width: "100%", height: 48 }} />
        <p className="hint">VRAM use over the last few minutes.</p>
      </div>

      <div className="card">
        <h2>Where each model is running</h2>
        <table className="control-table">
          <tbody>
            {g.models.map((m) => (
              <tr key={m.name}>
                <td className="who">
                  <div className="name">{m.name}</div>
                  <div className="hint">{m.detail}</div>
                </td>
                <td style={{ width: 190 }}>
                  {m.placement === "not loaded" || m.placement === "not running" || m.placement === "not started" ? (
                    <span className="hint">{m.placement}</span>
                  ) : (
                    <SplitBar gpu={m.gpu_percent} />
                  )}
                </td>
                <td className="hint">{m.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          The three models total roughly 5.5&nbsp;GB, so they do not fit at once on a 4&nbsp;GB
          card. The pipeline time-shares instead: the layout model and embedder while ingesting,
          the answering model while you ask questions.
        </p>
      </div>
    </div>
  );
}
