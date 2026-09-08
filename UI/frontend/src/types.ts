export type Source = {
  n: number;
  kind: "paper" | "chat" | "web";
  label: string;
  text?: string;
  distance?: number | null;
  filename?: string;
  title?: string;
  section?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  url?: string;
  chat_id?: string;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  warnings?: string[];
  error?: string;
  streaming?: boolean;
};

export type Status = {
  backend: "local" | "openai";
  ollama: boolean;
  model: string;
  model_pulled: boolean;
  openai_configured: boolean;
  openai_model: string;
  embeddings: boolean;
  registry: boolean;
};

export type ChatSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  embedded: boolean;
  n_messages: number;
};

export type ChatDetail = ChatSummary & {
  messages: { role: "user" | "assistant"; content: string; sources?: Source[] | null; created_at: string }[];
};

export type Settings = {
  llm: {
    backend: "local" | "openai";
    local: { url: string; model: string; num_ctx: number; keep_alive: string; temperature: number };
    openai: { base_url: string; api_key: string; model: string; temperature: number; max_tokens: number; stream: boolean };
  };
  retrieval: {
    n_results: number;
    n_results_with_web: number;
    use_chats: boolean;
    n_chat_results: number;
    web_results: number;
    web_chars_per_page: number;
  };
};

export type Ingestion = {
  services: Record<string, { pid: number; running: boolean }>;
  registry: null | {
    counts: Record<string, number>;
    total: number;
    stage_seconds: Record<string, number | null>;
    errors: { filename: string; error_count: number; last_error: string | null }[];
  };
  remaining?: number;
  eta_seconds: number | null;
  files: Record<string, number>;
};

export type QueryEvent =
  | { type: "chat"; chat_id: string }
  | { type: "sources"; sources: Source[]; backend: string }
  | { type: "warning"; message: string }
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };
