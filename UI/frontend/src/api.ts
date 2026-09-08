import type { ChatDetail, ChatSummary, Ingestion, QueryEvent, Settings, Status } from "./types";

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
  papers: () => json<{ papers: string[] }>("/v1/papers"),
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
  embedChat: (id: string) => json<{ embedded: number }>(`/v1/chats/${id}/embed`, { method: "POST" }),
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
