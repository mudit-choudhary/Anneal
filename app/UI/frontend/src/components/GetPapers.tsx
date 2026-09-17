import { useEffect, useState } from "react";
import { api } from "../api";
import type { ArxivCandidate, Settings } from "../types";

function ago(iso?: string | null) {
  if (!iso) return "";
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  if (days < 365) return `${Math.floor(days / 30)} months ago`;
  return `${Math.floor(days / 365)} years ago`;
}

/** Fetch new papers.
 *
 *  "Fetch 10 papers on X" is a blind command — you find out what arrived after
 *  it has been parsed and embedded. This looks first: the same arXiv search the
 *  downloader runs, shown as readable cards with abstracts, with anything
 *  already in the corpus marked. You then choose. A direct link still works for
 *  the case where you already know exactly what you want. */
export default function GetPapers({ onStarted }: { onStarted: () => void }) {
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState(10);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState<"" | "look" | "get" | "url">("");
  const [msg, setMsg] = useState("");
  const [found, setFound] = useState<ArxivCandidate[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    api.settings().then((s: Settings) => {
      setTopic(s.ingestion?.domain ?? "");
      setCount(s.ingestion?.max_papers ?? 10);
    }).catch(() => undefined);
  }, []);

  async function look() {
    setBusy("look");
    setMsg("");
    setFound(null);
    try {
      const r = await api.previewArxiv(topic, count);
      setFound(r.papers);
      // preselect everything not already held — the common case is "take them all"
      setPicked(new Set(r.papers.filter((p) => !p.already_have).map((p) => p.url)));
      api.saveSettings({ ingestion: { domain: topic, max_papers: count } } as Partial<Settings>)
        .catch(() => undefined);
      if (r.papers.length === 0) setMsg("arXiv returned nothing for that topic.");
    } catch (e) {
      setMsg(`Search failed: ${(e as Error).message}`);
    } finally {
      setBusy("");
    }
  }

  async function download() {
    const urls = [...picked];
    if (!urls.length) return;
    setBusy("get");
    try {
      await api.pickPapers(urls);
      setMsg(`Downloading ${urls.length} paper(s). They appear below as they register, then move through the pipeline.`);
      setFound(null);
      setPicked(new Set());
      onStarted();
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy("");
    }
  }

  async function byLink() {
    setBusy("url");
    try {
      await api.fetchUrl(url);
      setMsg("Downloading. It will appear below once registered.");
      setUrl("");
      onStarted();
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy("");
    }
  }

  const fresh = found?.filter((p) => !p.already_have).length ?? 0;
  const dupes = (found?.length ?? 0) - fresh;

  return (
    <div className="card getpapers">
      <h2>Get papers</h2>

      <div className="search">
        <input className="topic" value={topic} placeholder="Search arXiv — e.g. Retrieval Augmented Generation"
               onChange={(e) => setTopic(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && topic.trim() && look()} />
        <input className="n" type="number" min={1} max={50} value={count}
               onChange={(e) => setCount(Number(e.target.value))} />
        <button className="send" disabled={!topic.trim() || busy !== ""} onClick={look}>
          {busy === "look" ? "Searching…" : "Look first"}
        </button>
      </div>

      {found && found.length > 0 && (
        <>
          <div className="found-head">
            <span>
              <b>{fresh}</b> new{dupes > 0 && <span className="hint"> · {dupes} already here</span>}
            </span>
            <span className="spacer" />
            <button className="small" onClick={() => setPicked(new Set(found.filter((p) => !p.already_have).map((p) => p.url)))}>
              Select new
            </button>
            <button className="small" onClick={() => setPicked(new Set())}>Clear</button>
          </div>

          <div className="results">
            {found.map((p) => {
              const on = picked.has(p.url);
              return (
                <div key={p.url} className={`result ${on ? "on" : ""} ${p.already_have ? "have" : ""}`}
                     onClick={() => {
                       if (p.already_have && !on) return;
                       setPicked((cur) => {
                         const next = new Set(cur);
                         next.has(p.url) ? next.delete(p.url) : next.add(p.url);
                         return next;
                       });
                     }}>
                  <input type="checkbox" checked={on} readOnly tabIndex={-1} />
                  <div className="who">
                    <div className="t">{p.title}</div>
                    <div className="m">
                      {ago(p.published)}
                      {p.authors.length > 0 && ` · ${p.authors[0]}${p.authors.length > 1 ? " et al." : ""}`}
                      {p.categories.length > 0 && ` · ${p.categories[0]}`}
                      {p.already_have && <span className="tag">already here</span>}
                    </div>
                    {open === p.url && <p className="abs">{p.summary}…</p>}
                  </div>
                  <button className="x" title="Abstract"
                          onClick={(e) => { e.stopPropagation(); setOpen(open === p.url ? null : p.url); }}>
                    {open === p.url ? "−" : "+"}
                  </button>
                </div>
              );
            })}
          </div>

          <div className="row" style={{ marginTop: 12 }}>
            <button className="send" disabled={picked.size === 0 || busy !== ""} onClick={download}>
              {busy === "get" ? "Starting…" : `Download ${picked.size} paper${picked.size === 1 ? "" : "s"}`}
            </button>
            <button className="small" onClick={() => { setFound(null); setPicked(new Set()); }}>
              Discard results
            </button>
          </div>
        </>
      )}

      <div className="search" style={{ marginTop: 14 }}>
        <input className="topic" value={url}
               placeholder="Or paste a link — https://arxiv.org/abs/2401.01234"
               onChange={(e) => setUrl(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && url.trim() && byLink()} />
        <button className="send" disabled={!url.trim() || busy !== ""} onClick={byLink}>
          {busy === "url" ? "Starting…" : "Download"}
        </button>
      </div>

      <p className="hint">
        Downloading only registers a paper — <b>the parser must be running</b> for it to be
        processed. Start it from Pipeline control above.
      </p>
      {msg && <p className="hint">{msg}</p>}
    </div>
  );
}
