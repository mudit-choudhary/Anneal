import { useEffect, useState } from "react";
import { api } from "../api";
import type { Settings } from "../types";

/** Fetch new papers into the pipeline: N from an arXiv topic, or one by link. */
export default function GetPapers({ onStarted }: { onStarted: () => void }) {
  const [domain, setDomain] = useState("");
  const [count, setCount] = useState(10);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState<"" | "arxiv" | "url">("");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    api.settings().then((s: Settings) => {
      setDomain(s.ingestion?.domain ?? "");
      setCount(s.ingestion?.max_papers ?? 10);
    }).catch(() => undefined);
  }, []);

  async function go(kind: "arxiv" | "url") {
    setBusy(kind);
    setMsg("");
    try {
      if (kind === "arxiv") {
        await api.fetchArxiv(domain, count);
        // remember the topic and count for next time
        api.saveSettings({ ingestion: { domain, max_papers: count } } as Partial<Settings>).catch(() => undefined);
        setMsg(`Fetching up to ${count} paper(s) on "${domain}". Watch the download log below; ` +
               `they appear as "downloaded" then move through the pipeline.`);
      } else {
        await api.fetchUrl(url);
        setMsg("Downloading. It will appear below once registered.");
        setUrl("");
      }
      onStarted();
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="card">
      <h2>Get papers</h2>

      <div className="grid narrow">
        <label>arXiv topic</label>
        <input value={domain} placeholder="e.g. Retrieval Augmented Generation"
               onChange={(e) => setDomain(e.target.value)} />
        <label>How many</label>
        <input type="number" min={1} max={200} value={count}
               onChange={(e) => setCount(Number(e.target.value))} />
        <span />
        <button className="send" disabled={!domain.trim() || busy !== ""} onClick={() => go("arxiv")}>
          {busy === "arxiv" ? "Starting…" : `Fetch ${count} paper(s)`}
        </button>
      </div>

      <div className="grid narrow" style={{ marginTop: 14 }}>
        <label>Direct link</label>
        <input value={url} placeholder="https://arxiv.org/abs/2401.01234  or  https://host/paper.pdf"
               onChange={(e) => setUrl(e.target.value)} />
        <span />
        <button className="send" disabled={!url.trim() || busy !== ""} onClick={() => go("url")}>
          {busy === "url" ? "Starting…" : "Download"}
        </button>
      </div>

      <p className="hint">
        Newest papers are fetched first, skipping any already known. Downloading only registers
        them — <b>parse_manager must be running</b> for them to be processed
        (<code>scripts/start_query.sh --with-ingest</code>).
      </p>
      {msg && <p className="hint">{msg}</p>}
    </div>
  );
}
