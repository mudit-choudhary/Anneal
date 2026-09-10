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

The app has five tabs: **Chat**, **Ingestion**, **GPU**, **Logs**, **Settings**,
and a light/dark toggle in the top-right corner (remembered per browser).

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
- **Follow-up questions keep their context.** Asking "what does CELP do?"
  and then "give me the gist of this paper" works: the previous turns are
  sent to the model, and the previous question is folded into the search so
  retrieval stays on the same paper. The last 4 messages are carried (older
  assistant answers are truncated) to protect the context window — start a
  **+ New** chat when you change subject, so an old topic does not bias
  retrieval.
- Each answer shows **how long it took** and the time it was sent; both are
  stored, so they are still there when you reopen the chat.
- **Left rail — Chats**: grouped by day, newest first, with the time and
  message count. ✎ renames a chat, × deletes it, **+ New** starts a fresh
  subject. Every conversation is saved automatically as you ask.
- **Right rail — Filter papers**: grouped by the day each paper was added,
  with per-day *all/none* and collapsible sections. Search, *Select
  all/matching*, *Clear*. Nothing selected = search all.
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

**Ingestion** — a **Pipeline control** pane plus two action cards:

- **Pipeline control** — start, stop and restart every service (registry,
  parser, embedder, pruner, downloader) with a switch each, no terminal
  needed. Two presets do the common setups: *Set up for questions*
  (registry + embedder on CPU, GPU free for the model) and *Set up for adding
  papers* (also parser + pruner, embedder on GPU). The embedder's device is
  fixed at startup, so there is a restart-on-CPU/GPU control next to it.
  The web UI is the one service you cannot stop from here — it would kill the
  page; use `scripts/stop_services.sh`.


- **Get papers** — fetch N papers on an arXiv topic, or download one by
  direct link (arXiv `abs`/`pdf` links keep their real title). Newest first,
  skipping anything already known. This only *downloads and registers* them;
  **parse_manager must be running** for them to be processed, so start with
  `scripts/start_query.sh --with-ingest` on a day you plan to add papers.
- **Pruning** — what happens to raw PDFs once a paper is embedded: keep
  (default), archive to a folder, or delete, with a keep-newest/oldest-N
  option. **Preview a sweep** shows exactly what would be removed before
  anything is touched. Parsed and processed intermediates are always
  removed; raw PDFs are kept by default because they are the only input a
  re-ingest can rebuild from.
- Above them: how many papers are searchable, how many are still in the
  pipeline with their **total size**, and an **ETA measured from actual
  throughput**. If nothing is running to do the work, it says so instead of
  guessing — e.g. *"not progressing — parse is not running"*.
- Below: a stage strip (downloaded → parsed → processed → embedded → error)
  and **Files on disk**. Those two measure different things — the stages say
  where each paper *is*, the files are what is still on disk waiting for the
  pruner. `processed` shows twice as many files as papers because each paper
  writes both a readable `.txt` and the `.json` blocks the embedder reads.

**GPU** — VRAM used and free, a bar segmented by process (so you can see
exactly what is holding the card), utilisation and temperature, a short usage
history, and where each of the three models is actually running. It warns when
VRAM is nearly gone, since the next model to start will silently fall back to
the CPU.

**Logs** — live tail of any service, with a level filter (info/warning/error),
a text filter, and follow-on-scroll. Pick `download` to watch a fetch, or
`parse` during ingestion.

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
| Morning start | `scripts/start_query.sh` — or start services from the Ingestion tab |
| Check the whole pipeline works | `python scripts/smoke_test.py` |
| Rebuild the web UI | `cd UI/frontend && npm run build` (dev server: `npm run dev`) |
| Start incl. ingestion | `scripts/start_query.sh --with-ingest` |
| Add hand-copied PDFs | `python scripts/register_pdfs.py` |
| Retry errored papers | `python scripts/register_pdfs.py --retry-errors` |
| Download by link / topic | `python download_manager/downloader.py --url … / --once --domain … --max N` |
| Progress / health | Ingestion tab, or `python scripts/pipeline_status.py` |
| Debug an answer | `python scripts/rag_inspect.py retrieve\|answer "…"` |
| Stop everything | `scripts/stop_services.sh` |
| Rebuild from scratch | `scripts/fresh_start.sh --yes` — see [FRESH_START.md](FRESH_START.md) |
| Rebuild the web UI after frontend changes | `cd UI/frontend && npm run build` |
