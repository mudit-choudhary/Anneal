# Fresh Start — purge everything and re-ingest

Use this whenever the pipeline's output would change for papers already
ingested: a parser fix, a chunking change, a new embedding model, or a
registry schema change. Everything downstream of the raw PDFs is rebuilt.

```bash
scripts/fresh_start.sh            # previews what will be deleted, asks to confirm
scripts/fresh_start.sh --yes      # unattended
```

Raw PDFs in `data/raw_pdfs/` are **never** deleted.

## What the script does

| Step | Action | Why |
|---|---|---|
| 1 | `scripts/stop_services.sh` | services hold the DBs open |
| 2 | `scripts/reset_ingestion.py --yes` | deletes `vector_db/`, `registry_manager/rag_registry.db`, `data/parsed/*`, `data/processed/*` |
| 3 | start **registry** (:4000), wait until it answers | it recreates the DB with the *current* schema — the reason the DB file is deleted rather than reset |
| 4 | `scripts/register_pdfs.py` | inserts every PDF as status `downloaded`; without this, the parse loop has nothing to pick up |
| 5 | start **parse_manager**, **embedding_manager**, **prune_manager**, **UI** (:4002) | the pipeline runs on its own from here |

Services run in the background; logs go to `run/logs/<service>.log`, PIDs
to `run/pids/`. The embedding model (`bge-base`, ~440MB) downloads on the
first embedding-service start.

## Watching it run

```bash
python scripts/pipeline_status.py             # snapshot
watch -n 30 python scripts/pipeline_status.py # live
tail -f run/logs/parse.log run/logs/embedding.log
```

Papers move `downloaded → parsed → processed → embedded`. Rough throughput on
the GTX 1650: ~5s per paper for layout detection (both stages), a few
seconds per paper for embedding. Expect a 50-paper corpus to be fully
embedded well within an hour; the embedding loop polls every 60s so there
is some lag between stages.

The UI is usable as soon as the first paper reaches `embedded`.

## Unattended overnight run (auto-shutdown when done)

`scripts/wait_for_ingestion.py` blocks until every registered paper is
`embedded` (or `error`), so a shutdown can be chained after it without
touching `fresh_start.sh`:

```bash
scripts/fresh_start.sh --yes --no-ui
python scripts/wait_for_ingestion.py && scripts/stop_services.sh && systemctl poweroff
```

`systemctl poweroff` works without a password from a logged-in desktop
session; from SSH use `sudo sh -c 'python scripts/wait_for_ingestion.py && scripts/stop_services.sh && shutdown -h now'`
(one `sudo` up front, so the cached credential can't expire mid-wait).

The wait script exits non-zero — and the `&&` chain stops short of powering
off — if a service dies or nothing progresses for 30 minutes, so a stuck run
leaves the machine on for you to inspect `run/logs/`. Use `;` instead of
`&&` to power off regardless. Check progress from another terminal with
`python scripts/wait_for_ingestion.py --once` or `pipeline_status.py`.

## Stopping

```bash
scripts/stop_services.sh
```

Stopping mid-run is safe: every stage is idempotent and status-driven, so
restarting the services (or re-running steps 3–5 by hand) resumes where it
left off. Only a *fresh start* (steps 1–2) throws work away.

## Smoke test before a full run

```bash
scripts/fresh_start.sh --yes --limit 3      # only the first 3 PDFs are registered
python scripts/pipeline_status.py           # wait for embedded = 3
python scripts/rag_inspect.py retrieve "some question"
python scripts/register_pdfs.py             # then register the rest — no purge needed
```

The last line is the key trick: registering more PDFs into a running
pipeline just queues them. You never need a purge to *add* papers.

## Day-2 operations (no purge needed)

| I want to… | Do |
|---|---|
| Add PDFs I copied in by hand | `python scripts/register_pdfs.py` |
| Pull new papers from arXiv | `cd download_manager && python downloader.py` (registers what it downloads; capped per domain by `MAX_PAPERS_PER_DOMAIN`) |
| Re-parse one paper after a parser tweak | `python parse_manager/pdf_parser.py Paper.pdf && python parse_manager/txt_processor.py Paper` — the embedding loop re-embeds it when its status returns to `processed`; re-embedding replaces that paper's old chunks |
| Free VRAM for daytime querying | services keep running; Ollama unloads itself after 30 min idle; restart the embedding service with `EMBED_DEVICE=cpu` if you want its ~1GB back too |

## Manual equivalent

If you prefer to run the steps yourself (e.g. in separate terminals so you
see the logs live):

```bash
scripts/stop_services.sh
python scripts/reset_ingestion.py --yes
cd registry_manager  && python main.py &      # wait for "Uvicorn running"
python scripts/register_pdfs.py
cd parse_manager     && python main.py &
cd embedding_manager && python main.py &
cd prune_manager     && python pruning.py &
cd UI                && python main.py &
```
