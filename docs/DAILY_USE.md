# Daily Use — asking questions

Everything is ingested; the morning routine is one command.

## 1. Start

```bash
cd ~/Desktop/RAGSetup
scripts/start_query.sh          # = python scripts/ops.py start-query
```

Starts the registry (or reuses one already running), the embedding service
**on CPU** (so all 4GB of GPU stay free for Qwen), and the UI, unloads any
warm Ollama model so it reloads onto a free GPU, and waits until the
embedding model is loaded (~20s). It ends with `Ready — N papers in the
vector store` and the UI address. Ollama is a system service that starts on
boot; the script tells you if it isn't running.

## 2. Ask — http://127.0.0.1:4002

The app has three tabs: **Chat**, **Ingestion**, **Settings**.

**Chat**
- Header dots: answering model (Ollama or the configured API), Embeddings,
  Registry. Green = ready; amber on Ollama = model not pulled.
- The **first question of the day takes 10–20s** while Qwen loads; after that
  answers stream at ~15–20 tokens/s.
- Every claim carries a citation `[n]`; click it to open the exact excerpt
  (paper › section · page) in the Sources panel. Web pages and previous
  conversations are tagged in the panel.
- **Web** toggle next to Send: also fetches the top web pages for the
  question (DuckDuckGo + main-text extraction) and hands them to the model
  as extra excerpts. They are never embedded; paper excerpts are preferred
  when they disagree.
- Left sidebar: **Chats** (every conversation is saved automatically;
  click to reopen, × to delete) and **Filter papers** (restrict a question
  to selected papers; nothing selected = all).
- **Save chat to memory** (under the composer) embeds the chat's
  question/answer pairs so future questions can retrieve them. Saved chats
  are searched as a separate, lower-priority source; toggle this in
  Settings.
- Answers containing ```mermaid``` blocks render as diagrams; tables render
  as tables.

**Settings** — the answering model: `local` (Ollama URL, model, context,
temperature) or `openai` (any OpenAI-compatible API: base URL, key, model,
max tokens, streaming). Retrieval knobs: paper chunks per question, saved
chats on/off, web pages per question. Saved to `data/settings.json`; takes
effect on the next question.

**Ingestion** — progress bar, counts per status, ETA, average time per
stage, errors with their messages, which services are running, files on
disk, and live logs per service (verbose, Linux-style).

Prefer the terminal? `cd rag_setup && python rag.py`.

## 3. When an answer looks wrong

Find out *which stage* is at fault before blaming the model:

```bash
python scripts/rag_inspect.py retrieve "your question"           # right chunks in the top 8?
python scripts/rag_inspect.py answer   "your question" --web     # same path as the UI, sources printed first
```

If retrieval is wrong, the fix is upstream (parsing/chunking — see
[USER_GUIDE.md](USER_GUIDE.md)); if retrieval is right and the answer
isn't, try the other backend from Settings.

## 4. Adding papers (no purge needed)

```bash
scripts/start_query.sh --with-ingest          # parse + prune too; embedder on the GPU
cp /path/to/new/*.pdf data/raw_pdfs/ && python scripts/register_pdfs.py
cd download_manager && python downloader.py --url https://arxiv.org/abs/2401.01234
cd download_manager && python downloader.py --once --domain "Graph Neural Networks" --max 5
```

Watch them reach `embedded` on the Ingestion tab. Or let the daily timer do
it (see [setup.md](setup.md#daily-automatic-ingestion-systemd-user-timer)).
Papers are searchable the moment they are embedded — no restart.

## 5. Stop (optional)

```bash
scripts/stop_services.sh
```

Idle services cost nothing, and Ollama unloads Qwen after 30 minutes on its
own.

## Quick reference

| Task | Command |
|---|---|
| Morning start | `scripts/start_query.sh` |
| Start incl. ingestion | `scripts/start_query.sh --with-ingest` |
| Add hand-copied PDFs | `python scripts/register_pdfs.py` |
| Retry errored papers | `python scripts/register_pdfs.py --retry-errors` |
| Download by link / topic | `python download_manager/downloader.py --url … / --once --domain … --max N` |
| Progress / health | Ingestion tab, or `python scripts/pipeline_status.py` |
| Debug an answer | `python scripts/rag_inspect.py retrieve\|answer "…"` |
| Stop everything | `scripts/stop_services.sh` |
| Rebuild from scratch | `scripts/fresh_start.sh --yes` — see [FRESH_START.md](FRESH_START.md) |
| Rebuild the web UI after frontend changes | `cd UI/frontend && npm run build` |
