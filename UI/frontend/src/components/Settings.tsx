import { useEffect, useState } from "react";
import { api } from "../api";
import type { Settings } from "../types";

export default function SettingsPage({ onSaved }: { onSaved: () => void }) {
  const [s, setS] = useState<Settings | null>(null);
  const [msg, setMsg] = useState("");
  useEffect(() => {
    api.settings().then(setS).catch((e) => setMsg(String(e.message)));
  }, []);
  if (!s) return <div className="page">{msg || "Loading…"}</div>;

  const set = (path: string[], value: unknown) =>
    setS((cur) => {
      const next = structuredClone(cur!) as unknown as Record<string, unknown>;
      let d: Record<string, unknown> = next;
      for (const k of path.slice(0, -1)) d = d[k] as Record<string, unknown>;
      d[path[path.length - 1]] = value;
      return next as unknown as Settings;
    });

  async function save() {
    try {
      const saved = await api.saveSettings(s!);
      setS(saved);
      setMsg("Saved. Takes effect on the next question.");
      onSaved();
    } catch (e) {
      setMsg(`Save failed: ${(e as Error).message}`);
    }
  }

  const L = s.llm.local, O = s.llm.openai, R = s.retrieval;
  const num = (v: string) => Number(v);

  return (
    <div className="page settings">
      <h2>Answering model</h2>
      <div className="grid">
        <label>Backend</label>
        <select value={s.llm.backend} onChange={(e) => set(["llm", "backend"], e.target.value)}>
          <option value="local">local — Ollama on this machine</option>
          <option value="openai">openai — any OpenAI-compatible API (gateway, hosted model)</option>
        </select>
      </div>

      <fieldset disabled={s.llm.backend !== "local"}>
        <legend>Ollama (local)</legend>
        <div className="grid">
          <label>URL</label><input value={L.url} onChange={(e) => set(["llm", "local", "url"], e.target.value)} />
          <label>Model</label><input value={L.model} onChange={(e) => set(["llm", "local", "model"], e.target.value)} />
          <label>Context (num_ctx)</label><input type="number" value={L.num_ctx} onChange={(e) => set(["llm", "local", "num_ctx"], num(e.target.value))} />
          <label>Keep alive</label><input value={L.keep_alive} onChange={(e) => set(["llm", "local", "keep_alive"], e.target.value)} />
          <label>Temperature</label><input type="number" step="0.05" value={L.temperature} onChange={(e) => set(["llm", "local", "temperature"], num(e.target.value))} />
        </div>
        <p className="hint">On a 4 GB GPU, num_ctx is the VRAM knob: 6144 fits Qwen3-4B; larger spills to CPU.</p>
      </fieldset>

      <fieldset disabled={s.llm.backend !== "openai"}>
        <legend>OpenAI-compatible API</legend>
        <div className="grid">
          <label>Base URL</label><input placeholder="http://127.0.0.1:8080/v1" value={O.base_url} onChange={(e) => set(["llm", "openai", "base_url"], e.target.value)} />
          <label>API key</label><input type="password" value={O.api_key} onChange={(e) => set(["llm", "openai", "api_key"], e.target.value)} />
          <label>Model</label><input value={O.model} onChange={(e) => set(["llm", "openai", "model"], e.target.value)} />
          <label>Temperature</label><input type="number" step="0.05" value={O.temperature} onChange={(e) => set(["llm", "openai", "temperature"], num(e.target.value))} />
          <label>Max tokens</label><input type="number" value={O.max_tokens} onChange={(e) => set(["llm", "openai", "max_tokens"], num(e.target.value))} />
          <label>Stream</label>
          <label className="inline"><input type="checkbox" checked={O.stream} onChange={(e) => set(["llm", "openai", "stream"], e.target.checked)} /> stream tokens (falls back to a whole answer if the API refuses)</label>
        </div>
        <p className="hint">Retrieved excerpts are sent to whichever model is configured here. A masked key (••••) means “unchanged”.</p>
      </fieldset>

      <h2>Retrieval</h2>
      <div className="grid">
        <label>Paper chunks per question</label><input type="number" value={R.n_results} onChange={(e) => set(["retrieval", "n_results"], num(e.target.value))} />
        <label>… when web search is on</label><input type="number" value={R.n_results_with_web} onChange={(e) => set(["retrieval", "n_results_with_web"], num(e.target.value))} />
        <label>Search saved chats</label>
        <label className="inline"><input type="checkbox" checked={R.use_chats} onChange={(e) => set(["retrieval", "use_chats"], e.target.checked)} /> include previous conversations saved to memory</label>
        <label>Chat excerpts per question</label><input type="number" value={R.n_chat_results} onChange={(e) => set(["retrieval", "n_chat_results"], num(e.target.value))} />
        <label>Web pages per question</label><input type="number" value={R.web_results} onChange={(e) => set(["retrieval", "web_results"], num(e.target.value))} />
        <label>Chars per web page</label><input type="number" value={R.web_chars_per_page} onChange={(e) => set(["retrieval", "web_chars_per_page"], num(e.target.value))} />
      </div>

      <div className="row">
        <button className="send" onClick={save}>Save</button>
        <span className="hint">{msg}</span>
      </div>
    </div>
  );
}
