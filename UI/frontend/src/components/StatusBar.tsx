import type { Status } from "../types";

function Dot({ state, label, title }: { state: "ok" | "warn" | "bad"; label: string; title: string }) {
  return (
    <span title={title}>
      <i className={`dot ${state}`} />
      {label}
    </span>
  );
}

export default function StatusBar({ status }: { status: Status | null }) {
  if (!status) return <div className="status"><Dot state="bad" label="UI server" title="unreachable" /></div>;
  const llm =
    status.backend === "openai"
      ? { state: status.openai_configured ? "ok" : "warn", label: `API: ${status.openai_model || "not set"}`, title: "OpenAI-compatible backend" }
      : {
          state: status.ollama ? (status.model_pulled ? "ok" : "warn") : "bad",
          label: `Ollama: ${status.model}`,
          title: status.ollama ? (status.model_pulled ? "ready" : "model not pulled") : "daemon unreachable",
        };
  return (
    <div className="status">
      <Dot state={llm.state as "ok" | "warn" | "bad"} label={llm.label} title={llm.title} />
      <Dot state={status.embeddings ? "ok" : "bad"} label="Embeddings" title={status.embeddings ? "up" : "embedding service unreachable"} />
      <Dot state={status.registry ? "ok" : "warn"} label="Registry" title={status.registry ? "up" : "registry down (queries still work)"} />
    </div>
  );
}
