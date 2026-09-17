import { useEffect, useState } from "react";
import { api } from "../api";
import type { CoverageResult } from "../types";

/** Did every paper actually get parsed in full?
 *
 *  The registry records which stage a paper reached, never whether that stage
 *  covered the whole document — so a parse that died half way still ends up
 *  marked embedded, and the corpus quietly contains a paper that can answer
 *  about page one and nothing else. Two papers sat like that for days. This is
 *  the check, and the button does the repair that fixed them. */
export default function CoverageCard({ onRepaired }: { onRepaired?: () => void }) {
  const [r, setR] = useState<CoverageResult | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const load = () =>
    api.audit().then((res) => {
      setR(res);
      setPicked(new Set(res.papers.map((p) => p.filename)));
    }).catch((e) => setMsg(String(e.message)));

  useEffect(() => { load(); }, []);

  async function repair() {
    const filenames = [...picked];
    if (!filenames.length) return;
    setBusy(true);
    setMsg("");
    try {
      const res = await api.repair(filenames);
      setMsg(res.note);
      await load();
      onRepaired?.();
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  if (!r) return <div className="card"><h2>Coverage check</h2><p className="hint">{msg || "Checking…"}</p></div>;

  const clean = r.affected === 0;
  return (
    <div className={`card coverage ${clean ? "" : "flagged"}`}>
      <div className="row wrap">
        <h2>Coverage check</h2>
        <span className="spacer" />
        <button className="small" disabled={busy} onClick={load}>Re-check</button>
      </div>

      <p className={clean ? "verdict" : "verdict bad"}>
        {clean ? (
          <>All <b>{r.checked}</b> papers were parsed in full.</>
        ) : (
          <><b>{r.affected}</b> of {r.checked} papers were only partly parsed. They are
            searchable but incomplete.</>
        )}
      </p>

      {!clean && (
        <>
          <div className="results" style={{ maxHeight: 260 }}>
            {r.papers.map((p) => {
              const on = picked.has(p.filename);
              return (
                <div key={p.filename} className={`result ${on ? "on" : ""}`}
                     onClick={() => setPicked((cur) => {
                       const next = new Set(cur);
                       next.has(p.filename) ? next.delete(p.filename) : next.add(p.filename);
                       return next;
                     })}>
                  <input type="checkbox" checked={on} readOnly tabIndex={-1} />
                  <div className="who">
                    <div className="t">{p.filename}</div>
                    <div className="m">
                      {p.parsed_pages != null && p.pdf_pages != null && (
                        <b>{p.parsed_pages} of {p.pdf_pages} pages · </b>
                      )}
                      {p.problems.join(" · ")}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="row" style={{ marginTop: 12 }}>
            <button className="send" disabled={busy || picked.size === 0} onClick={repair}>
              {busy ? "Queuing…" : `Re-ingest ${picked.size} paper${picked.size === 1 ? "" : "s"}`}
            </button>
            <span className="hint">
              Deletes the partial output and sends them back through the pipeline from the PDF.
            </span>
          </div>
        </>
      )}

      {r.unverifiable > 0 && (
        <p className="hint">
          {r.unverifiable} paper{r.unverifiable === 1 ? " has" : "s have"} had the PDF pruned, so
          page coverage cannot be confirmed. Papers parsed from now on record their own page
          count and stay checkable.
        </p>
      )}
      {msg && <p className="hint">{msg}</p>}
    </div>
  );
}
