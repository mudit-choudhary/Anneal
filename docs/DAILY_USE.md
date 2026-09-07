# Daily Use — asking questions

Everything is already ingested; the morning routine is one command.

## 1. Start

```bash
cd ~/Desktop/RAGSetup
scripts/start_query.sh
```

This starts the registry, the embedding service **on CPU** (so all 4GB of
GPU stay free for Qwen), and the web UI, then waits until the embedding
model is loaded (~20s). It ends with `Ready — 46 papers in the vector store`
and the UI address. Ollama itself is a system service that starts on boot;
the script tells you if it isn't running (`sudo systemctl start ollama`).

## 2. Ask

Open **http://127.0.0.1:4002**.

- The two dots in the header must be green (Ollama, Embeddings). Amber on
  Ollama means the model isn't pulled; red means the daemon is down.
- The **first question of the day takes 10–20s** while Ollama loads Qwen
  into the GPU; after that answers stream at ~15–20 tokens/s.
- Every claim carries a citation like `[2]` — click it to see the exact
  chunk (paper › section · page) it came from. That's how you check the
  model isn't making things up.
- **Filter papers** in the left sidebar to restrict a question to specific
  papers; nothing selected searches everything.
- **Backend** dropdown: `local` is Qwen. `gemini` only works with
  `GEMINI_API_KEY` exported before starting the UI.

Prefer the terminal? `cd rag_setup && python rag.py`.

## 3. When an answer looks wrong

Find out *which stage* is at fault before blaming the model:

```bash
python scripts/rag_inspect.py retrieve "your question"   # are the right chunks in the top 8?
python scripts/rag_inspect.py answer   "your question"   # same answer path as the UI, sources printed first
```

If retrieval is wrong, the fix is upstream (parsing/chunking — see
[USER_GUIDE.md](USER_GUIDE.md)); if retrieval is right and the answer isn't,
try `--backend gemini` to see whether it's a 4B-model limit.

## 4. Adding papers (no purge needed)

Start with ingestion services too — the embedder then runs on the GPU:

```bash
scripts/start_query.sh --with-ingest
cp /path/to/new/*.pdf data/raw_pdfs/
python scripts/register_pdfs.py            # queues them; ~10s per paper end to end
python scripts/pipeline_status.py          # watch them reach "embedded"
```

Or pull new arXiv papers: `cd download_manager && python downloader.py`
(registers what it downloads; capped per domain). New papers are searchable
the moment they reach `embedded` — no restart.

Note that with `--with-ingest` the embedder holds ~1GB of GPU; Qwen still
fits (it spills ~10% to CPU). Switch back to plain `start_query.sh` once
you're done adding.

## 5. Stop (optional)

```bash
scripts/stop_services.sh
```

Idle services cost nothing, and Ollama unloads Qwen after 30 minutes on
its own — so leaving everything running is fine. Stop them if you need the
RAM/GPU for something else, or before a `fresh_start.sh`.

## If answers stream slowly

Check `ollama ps`. It should say `100% GPU`. A split like `73%/27% CPU/GPU`
means Qwen was loaded while something else held the GPU (typically the
embedder during ingestion) — Ollama keeps that placement as long as the
model stays warm. Fix: `ollama stop qwen3:4b-instruct`; the next question
reloads it onto the free GPU. `start_query.sh` does this automatically at
startup.

## Quick reference

| Task | Command |
|---|---|
| Morning start | `scripts/start_query.sh` |
| Start incl. ingestion | `scripts/start_query.sh --with-ingest` |
| Add hand-copied PDFs | `python scripts/register_pdfs.py` |
| Progress / health | `python scripts/pipeline_status.py` |
| Debug an answer | `python scripts/rag_inspect.py retrieve\|answer "…"` |
| Stop everything | `scripts/stop_services.sh` |
| Rebuild from scratch (after parser/chunker/model changes) | `scripts/fresh_start.sh --yes` — see [FRESH_START.md](FRESH_START.md) |
