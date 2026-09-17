# Operations — install, run, ingest, rebuild

> Paths here are relative to `app/`. The virtualenv and `tests/` sit at the repository root.

Everything needed to run Anneal day to day. Replaces the former
`setup.md`, `DAILY_USE.md` and `FRESH_START.md`.

## One command

```bash
anneal                # start everything and open the web app
anneal status         # what is running, what is indexed
anneal stop           # stop everything
```

`anneal` is `bin/anneal`, a wrapper around `scripts/ops.py` that uses the
project's virtualenv, so it works without activating anything. Put it on your
PATH once:

```bash
ln -s "$PWD/bin/anneal" ~/.local/bin/anneal    # run from app/
```

Bare `anneal` starts the registry, the embedding service **on CPU** (so all
4 GB of GPU stay free for the answering model), and the web app; it unloads any
warm Ollama model so it reloads onto a free GPU, waits for the embedding model
(~20 s), prints `Ready — N papers in the vector store`, and opens
<http://127.0.0.1:4002>. On the first run it builds the React app if
`UI/static` is missing.

| Flag / subcommand | Does |
|---|---|
| `anneal --with-ingest` | also runs parser + pruner, embedder on the GPU — use on a day you add papers |
| `anneal --no-open` | do not open a browser |
| `anneal status` | service table, Ollama, registry counts, vector-store size |
| `anneal stop [--all]` | stop services; `--all` also sweeps orphans by name and port |
| `anneal fresh-start [--yes] [--limit N]` | purge and re-ingest (below) |
| `anneal daily-ingest [--max N]` | one download cycle, process to completion, stop what it started |
| `anneal start\|restart <service>…` | one service at a time |

First build by hand, if you would rather not let `anneal` do it:

```bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"   # if node isn't on PATH
cd UI/frontend && npm install && npm run build
```

Without a build the UI service falls back to the legacy single-page
`UI/index.html` (chat only). `npm run dev` serves **only** the frontend, with no backend behind it; it is for
frontend development, where it proxies `/v1` to a running `anneal`.

## Requirements

- Linux, Python 3.12 (ops tooling is Python + psutil, so macOS/Windows are
  within reach; only the systemd timer is Linux-specific)
- NVIDIA GPU with ≥4 GB VRAM (GTX 1650 works; everything falls back to CPU)
- CUDA-enabled PyTorch
- Node 20+ (only to build the web UI)

```bash
python3.12 -m venv virtual_environments/annealenv                      # from the repository root
virtual_environments/annealenv/bin/pip install -r app/requirements.txt
virtual_environments/annealenv/bin/pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
```

`app/requirements.txt` lists exactly what the app imports and nothing else,
pinned to the versions the evaluation ran on. `tests/requirements.txt` adds
pytest, and `evals/requirements.txt` adds the harness's parsers and chunkers.
**Keep the last install line**: chromadb pulls in the CPU build of onnxruntime,
which overwrites the GPU build's files; reinstalling the GPU build last keeps
layout detection on the GPU. `anneal` uses the venv directly, so it never needs
activating.
Shared code lives in `common/` (paths, logging, registry client, settings
store, chat store).

## Ports and services

| Service | Port | Script | Needed for |
|---|---|---|---|
| registry_manager | 4000 | `registry_manager/main.py` | status tracking (all stages) |
| parse_manager | — | `parse_manager/main.py` | PDFs → layout JSON → tagged text |
| embedding_manager | 4001 | `embedding_manager/main.py` | embedding + retrieval (papers and saved chats) |
| prune_manager | — | `prune_manager/pruning.py` | deletes intermediates once embedded; raw PDFs kept or archived |
| download_manager | — | `download_manager/downloader.py` | new papers from arXiv or by URL |
| UI | 4002 | `UI/main.py` | web app (React) + JSON API |
| Ollama daemon | 11434 | systemd (automatic after install) | local LLM answers |

Logs go to `run/logs/<name>.log`, PIDs to `run/pids/`. The `.sh` files in
`scripts/` are thin wrappers kept for compatibility; `anneal` is the same code.

## Models

**Layout (YOLOv11)** — under `models/` (git-ignored). `MODEL_CANDIDATES` in
`parse_manager/config.py` is tried in order: `yolo11s_doc_layout_imgsz_1024` →
`yolo11_doc_layout_v2224_imgsz_1024` → `yolo11n_doc_layout_imgsz_1024`.
Inference loads the **`.onnx`** export, not the `.pt`; after copying in new
`.pt` weights, run the one-time export:

```bash
pip install ultralytics          # build-time only
python scripts/export_onnx.py
pip uninstall -y ultralytics     # optional; not a runtime dependency
```

ultralytics is AGPL-3.0 and is deliberately kept out of the runtime — see
[PENDING_IMPROVEMENTS.md](PENDING_IMPROVEMENTS.md) item 4 and
[PARSING.md](PARSING.md).

### GPU for layout detection

Layout inference uses the GPU only if **all three** hold: `onnxruntime-gpu` is
installed, CUDA 12.x / cuDNN 9.x are present, and the loader can find them.
Miss any one and onnxruntime **silently uses the CPU** — correct results,
roughly 6× slower, no error.

On this machine CUDA comes from the `nvidia-*-cu12` wheels that PyTorch pulls
in (`site-packages/nvidia/*/lib`), not a system install. The dynamic linker
does not search there, so `onnx_detector.py` preloads those libraries at
startup. The consequence: **if PyTorch is removed from this environment,
layout detection quietly drops to CPU** unless a system CUDA is installed.

```bash
grep "layout model on" run/logs/parse.log   # e.g. "layout model on CUDAExecutionProvider"
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

A `WARNING … not the GPU` line means the CUDA provider is installed but could
not load.

**Embedding** — `BAAI/bge-base-en-v1.5` (~440 MB) downloads from Hugging Face
into `~/.cache/huggingface/hub/` on the first embedding-service start; keep the
cache. **Docling** (optional parser trial) downloads its models on first use.
`python scripts/hardware_check.py` maps this machine's VRAM to model choices.

## Answering model

**Local (default)** — Qwen3-4B via Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b-instruct
```

Use the *instruct* (non-thinking) variant: the hybrid `qwen3:4b` ignores
Ollama's `think=false` and pollutes answers with chain-of-thought.

**Any OpenAI-compatible API** — open **Settings** in the app, choose backend
`openai`, enter base URL, key and model. Settings live in `data/settings.json`
(git-ignored). Environment variables seed the defaults: `LLM_BACKEND`,
`OLLAMA_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and
`EMBED_DEVICE` (`cuda` for overnight batch embedding, `cpu` during the day).

## The web app — http://127.0.0.1:4002

Five tabs — **Chat**, **Ingestion**, **GPU**, **Logs**, **Settings** — and a
light/dark toggle remembered per browser.

**Chat.** Header dots show the answering model, embeddings and registry; green
is ready, amber on Ollama means the model is not pulled. The **first question
of the day takes 10–20 s** while Qwen loads, then answers stream at ~15–20
tokens/s. Every claim carries a citation `[n]`; clicking it opens the exact
excerpt (paper › section · page). The **Web** toggle also fetches top web pages
(DuckDuckGo + main-text extraction) as extra excerpts — never embedded, and
paper excerpts win when they disagree. **Follow-ups keep context**: the last 4
messages are carried and the previous question folded into the search, so start
**+ New** when you change subject. Each answer shows how long it took. The left
rail lists chats by day (rename, delete, new); the right rail filters papers by
the day they were added. **Save chat to memory** embeds a conversation so later
questions can retrieve it. ```mermaid``` blocks render as diagrams.

**Ingestion.** A **Pipeline control** pane starts, stops and restarts every
service with a switch each, with two presets — *Set up for questions* (registry
+ embedder on CPU) and *Set up for adding papers* (also parser + pruner,
embedder on GPU). The embedder's device is fixed at startup, so there is a
restart-on-CPU/GPU control beside it. The web UI is the one service you cannot
stop from here — use `anneal stop`. **Get papers** fetches N papers on an arXiv
topic or one by direct link; this only downloads and registers, so the parser
must be running. **Pruning** decides what happens to raw PDFs once embedded:
keep (default), archive, or delete, with keep-newest/oldest-N; **Preview a
sweep** shows what would be removed first. Above them: how many papers are
searchable, how many are in flight with total size, and an **ETA measured from
actual throughput** — it says "not progressing — parse is not running" rather
than guessing. Below: the stage strip and **Files on disk** (these measure
different things; `processed` shows twice as many files as papers because each
writes a `.txt` and a `.json`).

**GPU.** VRAM used and free, a bar segmented by process, utilisation and
temperature, a short history, and where each model is actually running. It
warns when VRAM is nearly gone, since the next model to start silently falls
back to CPU.

**Logs.** Live tail of any service with level and text filters.

Prefer the terminal? `cd rag_setup && python rag.py`.

## Adding papers

```bash
anneal --with-ingest                                   # parser + pruner too, embedder on GPU
cp /path/to/new/*.pdf data/raw_pdfs/ && python scripts/register_pdfs.py
cd download_manager && python downloader.py --url https://arxiv.org/abs/2401.01234
cd download_manager && python downloader.py --once --domain "Graph Neural Networks" --max 5
```

Papers are searchable the moment they are embedded — no restart. Watch them
reach `embedded` on the Ingestion tab, or with `anneal status`.

### Daily automatic ingestion (systemd user timer)

Easiest from the app: **Ingestion → Daily downloads**. Add topics with a cap
each, pick a time, flip the switch. Only `loginctl enable-linger "$USER"` still
needs a terminal, because it needs root.

```bash
ops/systemd/install.sh --max 20 --time 03:00
systemctl --user enable --now rag-daily-ingest.timer
loginctl enable-linger "$USER"        # run while logged out too
```

Topics configured in the app live in `data/settings.json` under
`ingestion.schedule.topics`, and `anneal daily-ingest` searches each separately
so every topic carries its own cap. With no topics configured it falls back to
`download_manager/config.py`'s `DOMAINS` with a single shared cap. Changing
topics needs no reinstall; only the time of day is baked into the unit.

**Switching the timer on runs a cycle immediately** — `Persistent=true` treats
"never run" as a missed run, the same rule that fires a missed overnight run
five minutes after the next boot.

## Fresh start — purge and re-ingest

Use this whenever the pipeline's output would change for papers already
ingested: a parser fix, a chunking change, a new embedding model, or a registry
schema change.

```bash
anneal fresh-start          # previews what will be deleted, asks to confirm
anneal fresh-start --yes    # unattended
```

Raw PDFs in `data/raw_pdfs/` are **never** deleted.

| Step | Action | Why |
|---|---|---|
| 1 | stop all services (orphan sweep), unload Ollama models | services hold the DBs open; the GPU must be free for YOLO + the embedder |
| 2 | `scripts/reset_ingestion.py --yes` | deletes `vector_db/`, the registry DB, `data/parsed/*`, `data/processed/*` |
| 3 | start **registry** (:4000), wait until healthy | it recreates the DB with the *current* schema — why the file is deleted rather than reset |
| 4 | `scripts/register_pdfs.py` | inserts every PDF as `downloaded`; without it the parse loop has nothing to pick up |
| 5 | start **parse**, **embedding** (GPU), **prune**, **UI** | the pipeline runs on its own from here |

Papers move `downloaded → parsed → processed → embedded`; a failing stage sets
`error` with the message and is skipped until retried. Throughput on the
GTX 1650 is roughly 10 s per paper end to end.

**Unattended overnight run:**

```bash
anneal fresh-start --yes --no-ui
python scripts/wait_for_ingestion.py && anneal stop && systemctl poweroff
```

`wait_for_ingestion.py` exits 0 when every paper is `embedded` or `error`, 1 if
a service died, 2 if nothing progressed for 30 minutes — so a stuck run leaves
the machine on for you to inspect. Use `;` instead of `&&` to power off
regardless.

**Smoke test before a full run:**

```bash
anneal fresh-start --yes --limit 3      # only the first 3 PDFs are registered
python scripts/pipeline_status.py       # wait for embedded = 3
python scripts/rag_inspect.py retrieve "some question"
python scripts/register_pdfs.py         # then register the rest — no purge needed
```

Stopping mid-run is safe: every stage is idempotent and status-driven, so
restarting resumes where it left off. Only a *fresh start* throws work away.

## Day-2 operations (no purge needed)

| I want to… | Do |
|---|---|
| Add PDFs I copied in by hand | `python scripts/register_pdfs.py` |
| Pull new papers from arXiv | `python download_manager/downloader.py --once`, or the daily timer |
| Re-run one paper from scratch | `python scripts/register_pdfs.py --force Paper.pdf` |
| Retry papers that errored | `python scripts/register_pdfs.py --retry-errors` |
| Free VRAM for daytime querying | `anneal` (embedder on CPU); Ollama unloads itself after 30 min idle |
| Check the whole pipeline works | `python scripts/smoke_test.py` |
| Rebuild the web UI | `cd UI/frontend && npm run build` |

## When an answer looks wrong

Find out *which stage* is at fault before blaming the model:

```bash
watch -n 30 python scripts/pipeline_status.py                 # progress, refreshed
python scripts/rag_inspect.py retrieve "your question"        # right chunks in the top 8?
python scripts/rag_inspect.py answer   "your question" --web  # same path as the UI
```

If retrieval is wrong the fix is upstream (parsing or chunking — see
[USER_GUIDE.md](USER_GUIDE.md) and [PARSING.md](PARSING.md)); if retrieval is
right and the answer is not, try the other backend from Settings.

### Parsing or inspecting by hand

```bash
python parse_manager/pdf_parser.py SomePaper.pdf   # → data/parsed/SomePaper.json
python parse_manager/txt_processor.py SomePaper    # → data/processed/SomePaper.txt
python scripts/rag_inspect.py parse    SomePaper.pdf --pages 3
python scripts/rag_inspect.py tables   SomePaper.pdf
python scripts/rag_inspect.py chunks   SomePaper
python scripts/docling_compare.py SomePaper.pdf --docling-native
```

## Quick reference

| Task | Command |
|---|---|
| Morning start | `anneal` — or the Ingestion tab's *Set up for questions* |
| Start including ingestion | `anneal --with-ingest` |
| What is running / indexed | `anneal status` |
| Stop everything | `anneal stop` (`--all` to sweep orphans) |
| Check the whole pipeline works | `python scripts/smoke_test.py` |
| Add hand-copied PDFs | `python scripts/register_pdfs.py` |
| Retry errored papers | `python scripts/register_pdfs.py --retry-errors` |
| Download by link / topic | `python download_manager/downloader.py --url … / --once --domain … --max N` |
| Progress / health | Ingestion tab, or `python scripts/pipeline_status.py` |
| Debug an answer | `python scripts/rag_inspect.py retrieve\|answer "…"` |
| Rebuild from scratch | `anneal fresh-start --yes` |
| Rebuild the web UI after frontend changes | `cd UI/frontend && npm run build` |

## Tests

```bash
virtual_environments/annealenv/bin/pip install -r tests/requirements.txt   # from the repository root
virtual_environments/annealenv/bin/python -m pytest tests/ -q
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: no such file: 'X.pdf'` | Path was relative to your shell. Use a bare filename (resolved to `data/raw_pdfs/`) or an absolute path. |
| `registry unreachable …` from a stage | Registry (:4000) not running. One-off CLI runs still write files; the loops wait. `anneal` reuses a running registry or starts one. |
| A paper shows status `error` | The exception is in `last_error` (Ingestion tab or `pipeline_status.py`). Fix, then `python scripts/register_pdfs.py --retry-errors`. |
| `CUDA out of memory` during parsing | Something else holds the GPU. Parsing continues on CPU, slower. |
| Parsing ~2 pages/s instead of 10–14 | The layout model is on CPU — see "GPU for layout detection". |
| `has no ONNX export` on startup | New `.pt` weights were copied in. Run `python scripts/export_onnx.py`. |
| `port 4000 is already in use` | A stale service holds it. `anneal stop --all` sweeps orphans by pid, name and port. |
| UI shows the old single-page chat | `UI/static/` is missing — `anneal` builds it, or `cd UI/frontend && npm run build`. |
| Ollama dot amber | Daemon up, configured model not pulled — `ollama pull <model>`. |
| Answers stream slowly | `ollama ps` shows a CPU/GPU split: the model loaded while the GPU was busy. `ollama stop <model>`; `anneal` does this automatically. |
| Retrieval returns nothing | Nothing embedded yet — run the ingestion pipeline. |
| Changed chunking / embedding model | Collection names are tied to the model (`embedding_manager/config.py`); run `anneal fresh-start --yes`. |
