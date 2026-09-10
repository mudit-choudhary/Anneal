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
  duration_ms?: number | null;
  created_at?: string;
};

export type Paper = { filename: string; embedded_at: string | null };

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
  messages: {
    role: "user" | "assistant"; content: string; sources?: Source[] | null;
    created_at: string; duration_ms?: number | null;
  }[];
};

export type Settings = {
  llm: {
    backend: "local" | "openai";
    local: { url: string; model: string; num_ctx: number; keep_alive: string; temperature: number };
    openai: { base_url: string; api_key: string; model: string; temperature: number; max_tokens: number; stream: boolean };
  };
  ingestion: { domain: string; max_papers: number };
  prune: {
    raw_pdf_policy: "keep" | "archive" | "delete";
    archive_dir: string;
    keep_strategy: "all" | "newest" | "oldest";
    keep_count: number;
    interval_seconds: number;
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

export type PruneCounts = {
  parsed: number;
  processed: number;
  raw_archived: number;
  raw_deleted: number;
};

export type Ingestion = {
  services: Record<string, { pid: number; running: boolean }>;
  eta_note: string | null;
  throughput_per_hour: number | null;
  pending_bytes: number | null;
  registry: null | {
    counts: Record<string, number>;
    total: number;
    stage_seconds: Record<string, number | null>;
    errors: { filename: string; error_count: number; last_error: string | null }[];
  };
  remaining?: number;
  eta_seconds: number | null;
  files: Record<string, { files: number; papers: number; bytes: number }>;
};

export type QueryEvent =
  | { type: "chat"; chat_id: string }
  | { type: "sources"; sources: Source[]; backend: string }
  | { type: "warning"; message: string }
  | { type: "delta"; text: string }
  | { type: "done"; duration_ms?: number }
  | { type: "error"; message: string };

export type Service = {
  name: string;
  title: string;
  purpose: string;
  port: number | null;
  running: boolean;
  pid: number | null;
  uptime_seconds: number | null;
  self: boolean;
  log_bytes: number;
};

export type GpuInfo = {
  available: boolean;
  device: null | {
    name: string; total_mb: number; used_mb: number; free_mb: number;
    utilisation_percent: number; temperature_c: number;
  };
  processes: { pid: number; used_mb: number; owner: string }[];
  models: {
    name: string; detail: string; placement: string; gpu_percent: number;
    note: string; splittable: boolean | "partial";
  }[];
};

export type PruneCandidate = {
  filename: string;
  downloaded_at: string | null;
  bytes: number;
};

export type ArxivCandidate = {
  title: string;
  authors: string[];
  published: string | null;
  summary: string;
  url: string;
  pdf_url: string;
  categories: string[];
  already_have: boolean;
};

export type ScheduleTopic = { topic: string; max_papers: number; enabled: boolean };

export type ScheduleState = {
  time: string;
  topics: ScheduleTopic[];
  installed: boolean;
  enabled: boolean;
  next_run: string | null;
  last_run: string | null;
  linger: boolean;
  note?: string | null;
};

export type CoveragePaper = {
  filename: string;
  parsed_pages?: number | null;
  pdf_pages?: number | null;
  blocks?: number;
  chars?: number;
  problems: string[];
  unverified?: string;
};

export type CoverageResult = {
  checked: number;
  affected: number;
  unverifiable: number;
  papers: CoveragePaper[];
};
