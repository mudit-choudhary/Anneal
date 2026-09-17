import { useMemo, useState } from "react";
import { dayLabel } from "../format";
import type { Paper } from "../types";

const prettify = (f: string) => f.replace(/\.txt$/, "").replace(/_/g, " ");

/** Right rail: restrict retrieval to selected papers, grouped by the day they
 *  were added. Nothing selected = search everything. */
export default function PaperFilter({
  papers, selected, setSelected, onRefresh,
}: {
  papers: Paper[];
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  onRefresh: () => void;
}) {
  const [q, setQ] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const groups = useMemo(() => {
    const shown = papers.filter((p) => p.filename.toLowerCase().includes(q.toLowerCase()));
    // newest day first; unknown dates last
    const sorted = [...shown].sort((a, b) => (b.embedded_at ?? "").localeCompare(a.embedded_at ?? ""));
    const out: { day: string; items: Paper[] }[] = [];
    for (const p of sorted) {
      const day = dayLabel(p.embedded_at);
      const last = out[out.length - 1];
      if (last && last.day === day) last.items.push(p);
      else out.push({ day, items: [p] });
    }
    return out;
  }, [papers, q]);

  const shownCount = groups.reduce((n, g) => n + g.items.length, 0);
  const toggle = (name: string) => {
    const next = new Set(selected);
    next.has(name) ? next.delete(name) : next.add(name);
    setSelected(next);
  };
  const toggleDay = (day: string) => {
    const next = new Set(collapsed);
    next.has(day) ? next.delete(day) : next.add(day);
    setCollapsed(next);
  };
  const selectGroup = (items: Paper[]) => {
    const next = new Set(selected);
    const all = items.every((p) => next.has(p.filename));
    items.forEach((p) => (all ? next.delete(p.filename) : next.add(p.filename)));
    setSelected(next);
  };

  return (
    <aside className="rail rail-right">
      <div className="row">
        <h2>Filter papers</h2>
        <span className="hint">{selected.size ? `${selected.size} selected` : "all"}</span>
      </div>
      <input type="search" placeholder="Search papers…" value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="row">
        <button className="small" onClick={() => setSelected(new Set())}
                disabled={selected.size === 0}>Clear</button>
        <button className="small" onClick={onRefresh}>Refresh</button>
        {shownCount > 0 && (
          <button className="small"
                  onClick={() => setSelected(new Set(groups.flatMap((g) => g.items.map((p) => p.filename))))}>
            Select {q ? "matching" : "all"}
          </button>
        )}
      </div>

      <div className="list">
        {papers.length === 0 && <span className="hint">No embedded papers found.</span>}
        {papers.length > 0 && shownCount === 0 && <span className="hint">No papers match “{q}”.</span>}
        {groups.map((g) => {
          const isCollapsed = collapsed.has(g.day);
          const allSelected = g.items.every((p) => selected.has(p.filename));
          return (
            <div key={g.day} className="group">
              <div className="group-head">
                <button className="chevron" onClick={() => toggleDay(g.day)}
                        title={isCollapsed ? "Expand" : "Collapse"}>
                  {isCollapsed ? "▸" : "▾"}
                </button>
                <span onClick={() => toggleDay(g.day)}>{g.day}</span>
                <span className="hint">{g.items.length}</span>
                <button className="small" onClick={() => selectGroup(g.items)}
                        title={allSelected ? "Deselect this day" : "Select this day"}>
                  {allSelected ? "none" : "all"}
                </button>
              </div>
              {!isCollapsed && g.items.map((p) => (
                <label key={p.filename}>
                  <input type="checkbox" checked={selected.has(p.filename)}
                         onChange={() => toggle(p.filename)} />
                  {prettify(p.filename)}
                </label>
              ))}
            </div>
          );
        })}
      </div>

      <span className="hint">
        {papers.length} paper{papers.length === 1 ? "" : "s"} embedded. Nothing selected = search all.
      </span>
    </aside>
  );
}
