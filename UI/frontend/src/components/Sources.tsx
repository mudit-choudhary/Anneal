import { useEffect, useRef, useState } from "react";
import type { Source } from "../types";

function where(s: Source) {
  if (s.kind === "web") return s.url;
  if (s.kind === "chat") return "saved conversation";
  const parts = [];
  if (s.section) parts.push(`› ${s.section}`);
  if (s.page_start != null)
    parts.push(s.page_end != null && s.page_end !== s.page_start ? `pp.${s.page_start + 1}–${s.page_end + 1}` : `p.${s.page_start + 1}`);
  return parts.join(" · ");
}

export default function Sources({ sources, highlight }: { sources: Source[]; highlight: number | null }) {
  const [open, setOpen] = useState(false);
  const refs = useRef<Record<number, HTMLDivElement | null>>({});
  useEffect(() => {
    if (highlight == null) return;
    setOpen(true);
    const el = refs.current[highlight];
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [highlight]);
  if (!sources.length) return null;
  return (
    <details className="sources" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Sources ({sources.length})</summary>
      {sources.map((s) => (
        <div key={s.n} ref={(el) => (refs.current[s.n] = el)} className={`source ${s.kind} ${highlight === s.n ? "hl" : ""}`}>
          <div className="src-head">
            [{s.n}] {s.kind === "paper" ? s.title || s.filename : s.kind === "web" ? s.title : s.label.replace("previous conversation: ", "")}
            <span className="hint"> {where(s)}</span>
            {s.distance != null && <span className="hint"> · d={s.distance.toFixed(3)}</span>}
            {s.kind !== "paper" && <span className={`tag ${s.kind}`}>{s.kind}</span>}
          </div>
          <div className="src-text">{s.text}</div>
        </div>
      ))}
    </details>
  );
}
