import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { PruneCandidate, PruneCounts, Settings } from "../types";

function bytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function when(iso?: string | null) {
  if (!iso) return "unknown date";
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/** What happens to files once a paper is embedded.
 *
 *  The old version of this card was three dropdowns and a paragraph explaining
 *  what they meant together. The rule is really a cut through a shelf of papers
 *  ordered by age, so the shelf is drawn and the cut is shown on it: every PDF
 *  is a tile, oldest on the left, and moving the boundary repaints which ones
 *  survive. Nothing here asks the server — the whole list arrives once. */
export default function PruneCard() {
  const [s, setS] = useState<Settings["prune"] | null>(null);
  const [papers, setPapers] = useState<PruneCandidate[]>([]);
  const [registryUp, setRegistryUp] = useState(true);
  const [preview, setPreview] = useState<PruneCounts | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.settings().then((all) => setS(all.prune)).catch((e) => setMsg(String(e.message)));
    api.pruneCandidates()
      .then((r) => { setPapers(r.papers); setRegistryUp(r.registry); })
      .catch(() => setRegistryUp(false));
  }, []);

  const set = (k: keyof Settings["prune"], v: unknown) => {
    setS((cur) => (cur ? ({ ...cur, [k]: v } as Settings["prune"]) : cur));
    setPreview(null);
  };

  // Which PDFs survive, worked out exactly the way prune_manager does it.
  const verdict = useMemo(() => {
    if (!s) return { spared: new Set<string>(), acted: [] as PruneCandidate[] };
    if (s.raw_pdf_policy === "keep" || (s.raw_pdf_policy === "archive" && !s.archive_dir.trim())) {
      return { spared: new Set(papers.map((p) => p.filename)), acted: [] };
    }
    const n = Math.max(0, Math.min(s.keep_count, papers.length));
    let keep: PruneCandidate[] = [];
    if (s.keep_strategy === "newest") keep = n ? papers.slice(papers.length - n) : [];
    else if (s.keep_strategy === "oldest") keep = papers.slice(0, n);
    const spared = new Set(keep.map((p) => p.filename));
    return { spared, acted: papers.filter((p) => !spared.has(p.filename)) };
  }, [s, papers]);

  if (!s) return <div className="card"><h2>Pruning</h2><p className="hint">{msg || "Loading…"}</p></div>;

  async function save() {
    try {
      const saved = await api.saveSettings({ prune: s! } as Partial<Settings>);
      setS(saved.prune);
      setMsg("Saved. The prune service picks this up on its next sweep.");
      return true;
    } catch (e) {
      setMsg(`Save failed: ${(e as Error).message}`);
      return false;
    }
  }

  // A preview uses what is on screen; a real run saves it first, so the
  // background service and this button never disagree.
  async function run(dryRun: boolean) {
    setBusy(true);
    setMsg("");
    try {
      if (!dryRun && !(await save())) return;
      const r = await api.prune(dryRun, s!);
      setPreview(r.counts);
      const total = Object.values(r.counts).reduce((a, b) => a + b, 0);
      setMsg(dryRun
        ? (total ? `Would act on ${total} file(s). Nothing changed.` : "Nothing to do.")
        : `Done: ${total} file(s).`);
      if (!dryRun) {
        const fresh = await api.pruneCandidates();
        setPapers(fresh.papers);
      }
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const policy = s.raw_pdf_policy;
  const acting = policy !== "keep";
  const freed = verdict.acted.reduce((a, p) => a + p.bytes, 0);
  const held = papers.reduce((a, p) => a + p.bytes, 0) - freed;
  const verb = policy === "delete" ? "deleted" : "archived";
  const noFolder = policy === "archive" && !s.archive_dir.trim();   // pruning.py keeps PDFs then

  return (
    <div className="card prune">
      <h2>Pruning</h2>
      <p className="hint">
        Parsed and processed intermediates are always removed once a paper is embedded.
        This decides what happens to the original PDF.
      </p>

      {/* the decision, as one sentence that changes as you set it */}
      <p className={`verdict ${policy === "delete" && verdict.acted.length ? "bad" : ""}`}>
        {noFolder ? (
          <>No archive folder set, so all <b>{papers.length}</b> PDFs stay on disk until you enter one.</>
        ) : !acting ? (
          <>All <b>{papers.length}</b> PDFs stay on disk.</>
        ) : verdict.acted.length === 0 ? (
          <>Nothing to {policy}. All <b>{papers.length}</b> are within the spared range.</>
        ) : (
          <><b>{verdict.acted.length}</b> of {papers.length} PDFs will be {verb},
            freeing <b>{bytes(freed)}</b>. {verdict.spared.size} stay ({bytes(held)}).</>
        )}
      </p>

      {/* the shelf: one tile per PDF, oldest left */}
      {papers.length > 0 ? (
        <>
          <div className="shelf" role="img"
               aria-label={`${papers.length} PDFs oldest to newest; ${verdict.spared.size} kept`}>
            {papers.map((p) => {
              const kept = verdict.spared.has(p.filename);
              return (
                <span key={p.filename}
                      className={`tile ${kept ? "kept" : policy === "delete" ? "gone" : "moved"}`}
                      title={`${p.filename}\n${when(p.downloaded_at)} · ${bytes(p.bytes)}\n${kept ? "kept" : verb}`} />
              );
            })}
          </div>
          <div className="shelf-axis">
            <span>oldest · {when(papers[0]?.downloaded_at)}</span>
            <span>newest · {when(papers[papers.length - 1]?.downloaded_at)}</span>
          </div>
        </>
      ) : (
        <p className="hint">
          {registryUp ? "No embedded papers have a PDF on disk yet." : "Registry is not running, so the shelf cannot be drawn."}
        </p>
      )}

      {/* what to do with them */}
      <div className="seg" role="group" aria-label="What happens to raw PDFs">
        {(["keep", "archive", "delete"] as const).map((k) => (
          <button key={k} className={policy === k ? "on" : ""} onClick={() => set("raw_pdf_policy", k)}>
            {k === "keep" ? "Keep everything" : k === "archive" ? "Move to archive" : "Delete"}
          </button>
        ))}
      </div>

      {acting && (
        <>
          <div className="seg small-seg" role="group" aria-label="Which PDFs are spared">
            {(["all", "newest", "oldest"] as const).map((k) => (
              <button key={k} className={s.keep_strategy === k ? "on" : ""}
                      onClick={() => set("keep_strategy", k)}>
                {k === "all" ? "Spare none" : k === "newest" ? "Spare newest" : "Spare oldest"}
              </button>
            ))}
          </div>

          {s.keep_strategy !== "all" && (
            <div className="spare">
              <label htmlFor="spare-n">Spare the {s.keep_strategy}</label>
              <input id="spare-n" type="range" min={0} max={papers.length || 100}
                     value={Math.min(s.keep_count, papers.length || 100)}
                     onChange={(e) => set("keep_count", Number(e.target.value))} />
              <input type="number" min={0} value={s.keep_count}
                     onChange={(e) => set("keep_count", Number(e.target.value))} />
              <span className="hint">of {papers.length}</span>
            </div>
          )}

          {policy === "archive" && (
            <div className="grid narrow">
              <label>Archive folder</label>
              <input value={s.archive_dir} placeholder="required, e.g. /media/you/Drive/paper_archive"
                     onChange={(e) => set("archive_dir", e.target.value)} />
            </div>
          )}
        </>
      )}

      <div className="grid narrow">
        <label>Sweep every</label>
        <span className="with-unit">
          <input type="number" min={60} step={60} value={s.interval_seconds}
                 onChange={(e) => set("interval_seconds", Number(e.target.value))} />
          <span className="unit">
            seconds{s.interval_seconds >= 60 && ` (${(s.interval_seconds / 60).toFixed(0)} min)`}
          </span>
        </span>
      </div>

      {policy === "delete" && verdict.acted.length > 0 && (
        <p className="warning">
          ⚠ Deleting raw PDFs is irreversible. They are the only input a re-ingest can rebuild
          from — after a parser, chunking or embedding-model change you would have to download
          them again. Archiving keeps that option.
        </p>
      )}

      <div className="row" style={{ marginTop: 14 }}>
        <button className="send" onClick={save}>Save</button>
        <button className="small" disabled={busy} onClick={() => run(true)}>Preview a sweep</button>
        <button className="small" disabled={busy || !preview} onClick={() => run(false)}>
          Run it now
        </button>
      </div>

      {preview && (
        <table className="services" style={{ marginTop: 10 }}>
          <tbody>
            {Object.entries(preview).map(([k, v]) => (
              <tr key={k}><td>{k.replace("_", " ")}</td><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
      )}
      {msg && <p className="hint">{msg}</p>}
    </div>
  );
}
