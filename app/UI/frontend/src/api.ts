import type { ArxivCandidate, CoverageResult, ChatDetail, ChatSummary, GpuInfo, Ingestion, Paper, PruneCandidate, PruneCounts, ScheduleState, ScheduleTopic, QueryEvent, Service, Settings, Status } from "./types";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json();
}

export const api = {
  status: () => json<Status>("/v1/status"),
  papers: () => json<{ papers: Paper[] }>("/v1/papers"),
  settings: () => json<Settings>("/v1/settings"),
  saveSettings: (update: Partial<Settings>) =>
    json<Settings>("/v1/settings", { method: "PUT", body: JSON.stringify(update) }),
  ingestion: () => json<Ingestion>("/v1/ingestion"),
  chats: () => json<{ chats: ChatSummary[] }>("/v1/chats"),
  chat: (id: string) => json<ChatDetail>(`/v1/chats/${id}`),
  createChat: (title: string) => json<ChatSummary>("/v1/chats", { method: "POST", body: JSON.stringify({ title }) }),
  renameChat: (id: string, title: string) =>
    json<{ success: boolean }>(`/v1/chats/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteChat: (id: string) => json<{ success: boolean }>(`/v1/chats/${id}`, { method: "DELETE" }),
  deleteChats: (what: { ids: string[] } | { scope: "all" | "unanswered" }) =>
    json<{ deleted: number; requested: number; note?: string }>("/v1/chats", {
      method: "DELETE", body: JSON.stringify(what),
    }),
  embedChat: (id: string) => json<{ embedded: number }>(`/v1/chats/${id}/embed`, { method: "POST" }),
  fetchArxiv: (domain: string, max_papers: number) =>
    json<{ started: boolean; pid: number }>("/v1/ingest/arxiv", {
      method: "POST", body: JSON.stringify({ domain, max_papers }),
    }),
  fetchUrl: (url: string) =>
    json<{ started: boolean; pid: number }>("/v1/ingest/url", {
      method: "POST", body: JSON.stringify({ url }),
    }),
  previewArxiv: (topic: string, max_papers: number) =>
    json<{ topic: string; papers: ArxivCandidate[] }>(
      `/v1/ingest/arxiv/preview?topic=${encodeURIComponent(topic)}&max_papers=${max_papers}`),
  pickPapers: (urls: string[]) =>
    json<{ started: boolean; count: number }>("/v1/ingest/pick", {
      method: "POST", body: JSON.stringify({ urls }),
    }),
  audit: () => json<CoverageResult>("/v1/audit"),
  repair: (filenames: string[]) =>
    json<{ repaired: string[]; skipped: { filename: string; why: string }[]; note: string }>(
      "/v1/audit/repair", { method: "POST", body: JSON.stringify({ filenames }) }),
  schedule: () => json<ScheduleState>("/v1/schedule"),
  saveSchedule: (body: { topics?: ScheduleTopic[]; time?: string; enabled?: boolean }) =>
    json<ScheduleState>("/v1/schedule", { method: "PUT", body: JSON.stringify(body) }),
  gpu: () => json<GpuInfo>("/v1/gpu"),
  services: () => json<{ services: Service[] }>("/v1/services"),
  controlService: (name: string, action: "start" | "stop" | "restart", embed_device?: string) =>
    json<{ name: string; running: boolean; note?: string; device?: string }>(`/v1/services/${name}`, {
      method: "POST", body: JSON.stringify({ action, embed_device }),
    }),
  servicePreset: (preset: "query" | "ingest") =>
    json<{ started: string[]; already_running: string[]; note: string | null }>(
      `/v1/services/preset/${preset}`, { method: "POST" }),
  pruneCandidates: () => json<{ papers: PruneCandidate[]; registry: boolean }>("/v1/prune/candidates"),
  prune: (dryRun: boolean) =>
    json<{ dry_run: boolean; counts: PruneCounts }>(`/v1/prune/run?dry_run=${dryRun}`, { method: "POST" }),
};

/** POST /v1/query and deliver each newline-delimited JSON event as it arrives. */
export async function streamQuery(
  body: { query: string; filenames?: string[] | null; web?: boolean; chat_id?: string | null },
  onEvent: (ev: QueryEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch("/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`${r.status}: ${r.statusText}`);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}

/** Server-sent events tail of run/logs/<service>.log. Returns a stop function. */
export function streamLogs(service: string, lines: number, onLine: (line: string) => void, onError?: () => void) {
  const es = new EventSource(`/v1/logs/stream?service=${encodeURIComponent(service)}&lines=${lines}`);
  es.onmessage = (e) => onLine(e.data);
  es.onerror = () => onError?.();
  return () => es.close();
}
