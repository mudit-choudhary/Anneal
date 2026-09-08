# Fresh Start — purge everything and re-ingest

Use this whenever the pipeline's output would change for papers already
ingested: a parser fix, a chunking change, a new embedding model, or a
registry schema change. Everything downstream of the raw PDFs is rebuilt.

```bash
scripts/fresh_start.sh            # previews what will be deleted, asks to confirm
scripts/fresh_start.sh --yes      # unattended
```

(`= python scripts/ops.py fresh-start`.) Raw PDFs in `data/raw_pdfs/` are
**never** deleted.

## What it does

| Step | Action | Why |
|---|---|---|
| 1 | stop all services (orphan sweep), unload Ollama models | services hold the DBs open; the GPU must be free for YOLO + the embedder |
| 2 | `scripts/reset_ingestion.py --yes` | deletes `vector_db/`, `registry_manager/rag_registry.db`, `data/parsed/*`, `data/processed/*` |
| 3 | start **registry** (:4000), wait until healthy | it recreates the DB with the *current* schema — the reason the file is deleted rather than reset |
| 4 | `scripts/register_pdfs.py` | inserts every PDF as status `downloaded`; without this the parse loop has nothing to pick up |
| 5 | start **parse**, **embedding** (GPU), **prune**, **UI** | the pipeline runs on its own from here |

Logs: `run/logs/<service>.log`. The embedding model downloads on the first
embedding-service start (~440MB).

## Watching it run

Ingestion tab in the UI (progress, ETA, errors, live logs), or:

```bash
python scripts/pipeline_status.py
watch -n 30 python scripts/pipeline_status.py
```

Papers move `downloaded → parsed → processed → embedded`; a stage that
fails sets `error` with the message, and that paper is skipped until you
retry it (`python scripts/register_pdfs.py --retry-errors`). Throughput on
the GTX 1650 is roughly 10s per paper end to end.

## Unattended overnight run (auto-shutdown when done)

```bash
scripts/fresh_start.sh --yes --no-ui
python scripts/wait_for_ingestion.py && scripts/stop_services.sh && systemctl poweroff
```

`wait_for_ingestion.py` exits 0 when every paper is `embedded` or `error`,
1 if a service died, 2 if nothing progressed for 30 minutes — so a stuck run
leaves the machine on for you to inspect. Use `;` instead of `&&` to power
off regardless.

## Stopping

```bash
scripts/stop_services.sh     # = ops.py stop --all
```

Stopping mid-run is safe: every stage is idempotent and status-driven, so
restarting resumes where it left off. Only a *fresh start* throws work away.

## Smoke test before a full run

```bash
scripts/fresh_start.sh --yes --limit 3      # only the first 3 PDFs are registered
python scripts/pipeline_status.py           # wait for embedded = 3
python scripts/rag_inspect.py retrieve "some question"
python scripts/register_pdfs.py             # then register the rest — no purge needed
```

## Day-2 operations (no purge needed)

| I want to… | Do |
|---|---|
| Add PDFs I copied in by hand | `python scripts/register_pdfs.py` |
| Pull new papers from arXiv | `python download_manager/downloader.py --once` (capped per domain), or the daily timer |
| Re-run one paper from scratch | `python scripts/register_pdfs.py --force Paper.pdf` — its old chunks are replaced when re-embedded |
| Retry papers that errored | `python scripts/register_pdfs.py --retry-errors` |
| Free VRAM for daytime querying | `scripts/start_query.sh` (embedder on CPU); Ollama unloads itself after 30 min idle |
