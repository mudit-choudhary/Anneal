import { useEffect, useState } from "react";
import { api } from "../api";
import type { PruneCounts, Settings } from "../types";

/** What happens to files once a paper is embedded. Intermediates are always
 *  removed; raw PDFs are kept by default because they are the only input a
 *  re-ingest can be rebuilt from. */
export default function PruneCard() {
  const [s, setS] = useState<Settings["prune"] | null>(null);
  const [preview, setPreview] = useState<PruneCounts | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.settings().then((all) => setS(all.prune)).catch((e) => setMsg(String(e.message)));
  }, []);
  if (!s) return <div className="card"><h2>Pruning</h2><p className="hint">{msg || "Loading…"}</p></div>;

  const set = (k: keyof Settings["prune"], v: unknown) => setS({ ...s, [k]: v } as Settings["prune"]);

  async function save() {
    try {
      const saved = await api.saveSettings({ prune: s! } as Partial<Settings>);
      setS(saved.prune);
      setMsg("Saved. The prune service picks this up on its next sweep.");
    } catch (e) {
      setMsg(`Save failed: ${(e as Error).message}`);
    }
  }

  async function run(dryRun: boolean) {
    setBusy(true);
    setMsg("");
    try {
      const r = await api.prune(dryRun);
      setPreview(r.counts);
      const total = Object.values(r.counts).reduce((a, b) => a + b, 0);
      setMsg(dryRun
        ? (total ? `Would remove ${total} file(s). Nothing changed.` : "Nothing to remove.")
        : `Removed ${total} file(s).`);
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const destructive = s.raw_pdf_policy === "delete";
  return (
    <div className="card">
      <h2>Pruning</h2>
      <p className="hint">
        Parsed and processed intermediates are always removed once a paper is embedded. This
        controls what happens to the original PDF.
      </p>
      {s.raw_pdf_policy !== "keep" && (
        <p className="hint">
          <b>In plain terms:</b> once a paper is searchable, its PDF is{" "}
          {s.raw_pdf_policy === "delete" ? "deleted" : "moved to the archive folder"}
          {s.keep_strategy === "all"
            ? "."
            : ` — except the ${s.keep_count} ${s.keep_strategy === "newest" ? "most recently added" : "oldest"} papers, whose PDFs stay in data/raw_pdfs.`}
        </p>
      )}

      <div className="grid narrow">
        <label>Raw PDFs</label>
        <select value={s.raw_pdf_policy} onChange={(e) => set("raw_pdf_policy", e.target.value)}>
          <option value="keep">keep — leave them in data/raw_pdfs (recommended)</option>
          <option value="archive">archive — move them elsewhere</option>
          <option value="delete">delete — remove permanently</option>
        </select>

        {s.raw_pdf_policy === "archive" && (
          <>
            <label>Archive folder</label>
            <input value={s.archive_dir} placeholder="/media/mudit/Drive/paper_archive"
                   onChange={(e) => set("archive_dir", e.target.value)} />
          </>
        )}

        {s.raw_pdf_policy !== "keep" && (
          <>
            <label>Apply to</label>
            <select value={s.keep_strategy} onChange={(e) => set("keep_strategy", e.target.value)}>
              <option value="all">every embedded paper</option>
              <option value="newest">all except the {s.keep_count} most recent</option>
              <option value="oldest">all except the {s.keep_count} oldest</option>
            </select>
            {s.keep_strategy !== "all" && (
              <>
                <label>Number to spare</label>
                <input type="number" min={0} value={s.keep_count}
                       onChange={(e) => set("keep_count", Number(e.target.value))} />
              </>
            )}
          </>
        )}

        <label>Sweep every</label>
        <span className="with-unit">
          <input type="number" min={60} step={60} value={s.interval_seconds}
                 onChange={(e) => set("interval_seconds", Number(e.target.value))} />
          <span className="unit">
            seconds{s.interval_seconds >= 60 && ` (${(s.interval_seconds / 60).toFixed(0)} min)`}
          </span>
        </span>
      </div>

      {destructive && (
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
