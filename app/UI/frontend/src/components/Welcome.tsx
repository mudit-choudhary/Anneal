import { useMemo } from "react";

// Annealing: heat, then cool slowly until the structure settles. The taglines
// lean on that, the way the chunker grows sections before they set.
const TAGLINES = [
  "Heat the question. Let the answer cool into shape.",
  "Slow-cooled answers, every claim traceable to a page.",
  "Dense papers in, clear structure out.",
  "Your library, tempered for questions.",
  "Every answer arrives with its receipts.",
];

const STARTERS = [
  { icon: "🌅", text: "What's new since yesterday?" },
  { icon: "🧭", text: "What are the main themes across my papers?" },
  { icon: "⚖️", text: "Compare the retrieval methods these papers propose, and where they disagree." },
  { icon: "🧪", text: "Which results would be hardest to reproduce, and why?" },
];

function greeting(hour: number) {
  if (hour < 5) return "Burning the midnight oil";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  if (hour < 22) return "Good evening";
  return "Late-night reading";
}

type Props = { paperCount: number; scoped: number | null; onPick: (q: string) => void };

export default function Welcome({ paperCount, scoped, onPick }: Props) {
  // picked once per mount, so the line doesn't change while you type
  const { hello, line } = useMemo(() => ({
    hello: greeting(new Date().getHours()),
    line: TAGLINES[Math.floor(Math.random() * TAGLINES.length)],
  }), []);

  return (
    <div className="welcome">
      <div className="glow" aria-hidden="true" />
      <p className="hello">{hello}.</p>
      <h2>{line}</h2>
      <p className="shelf-line">
        {scoped != null
          ? <>Searching <b>{scoped}</b> selected paper{scoped === 1 ? "" : "s"}.</>
          : paperCount > 0
            ? <><b>{paperCount}</b> papers are ready to answer you.</>
            : <>No papers embedded yet. Add some from the Ingestion tab.</>}
      </p>

      <div className="starters">
        {STARTERS.map((s) => (
          <button key={s.text} onClick={() => onPick(s.text)}>
            <span aria-hidden="true">{s.icon}</span>{s.text}
          </button>
        ))}
      </div>

      <p className="tips">
        Click a citation like <a className="cite">[1]</a> to read its excerpt · tick <b>Web</b> to add live pages ·
        Mermaid diagrams draw themselves
      </p>
    </div>
  );
}
