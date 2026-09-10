# Setup & Running

## Requirements

- Linux, Python 3.12 (the ops tooling is Python + psutil, so macOS/Windows
  are within reach; only the systemd timer is Linux-specific)
- NVIDIA GPU with ≥4GB VRAM (GTX 1650 works; everything falls back to CPU)
- CUDA-enabled PyTorch
- Node 20+ (only to build the web UI; installed via nvm in user space)

## Ports & services at a glance

| Service | Port | Script | Needed for |
|---|---|---|---|
| registry_manager | 4000 | `registry_manager/main.py` | status tracking (all pipeline stages) |
| parse_manager | — | `parse_manager/main.py` | PDFs → layout JSON → tagged text |
| embedding_manager | 4001 | `embedding_manager/main.py` | embedding + retrieval (papers and saved chats) |
| prune_manager | — | `prune_manager/pruning.py` | deletes parsed/processed intermediates once embedded; raw PDFs kept (or archived) |
| download_manager | — | `download_manager/downloader.py` | new papers from arXiv or by URL (manual or timer) |
| UI | 4002 | `UI/main.py` | web app (React) + JSON API |
| Ollama daemon | 11434 | systemd (automatic after install) | local LLM answers |

All service processes are managed by **`scripts/ops.py`** (`fresh-start`,
`start-query`, `stop`, `daily-ingest`, `start`, `restart`); the `.sh` files
in `scripts/` are thin wrappers around it. Logs go to `run/logs/<name>.log`,
PIDs to `run/pids/`.

## Environment

`globalragsetup_env` has the full dependency set:

```bash
source virtual_environments/globalragsetup_env/bin/activate
```

Per-service `requirements.txt` files list what each needs on its own.
Shared code lives in `common/` (paths, logging, registry client, settings
store, chat store).

## Models

- **Layout (YOLOv11)** — under `models/` (git-ignored). `MODEL_CANDIDATES` in
  `parse_manager/config.py` is tried in order: `yolo11s_doc_layout_imgsz_1024`
  → `yolo11_doc_layout_v2224_imgsz_1024` → `yolo11n_doc_layout_imgsz_1024`.
  Inference loads the **`.onnx`** export, not the `.pt`; after copying in new
  `.pt` weights, run the one-time export:

  ```bash
  pip install ultralytics          # build-time only
  python scripts/export_onnx.py
  pip uninstall -y ultralytics     # optional; it is not a runtime dependency
  ```

  ultralytics is AGPL-3.0 and is deliberately kept out of the runtime — see
  [PENDING_IMPROVEMENTS.md](PENDING_IMPROVEMENTS.md) item 4.

### GPU for layout detection

Layout inference uses the GPU only if **all three** hold: `onnxruntime-gpu`
is installed, CUDA 12.x / cuDNN 9.x are present, and the loader can find
them. Miss any one and onnxruntime **silently uses the CPU** — correct
results, roughly 6x slower, no error.

On this machine CUDA comes from the `nvidia-*-cu12` wheels that PyTorch
pulls in (`site-packages/nvidia/*/lib`), not from a system install. The
dynamic linker does not search there, so `onnx_detector.py` preloads those
libraries at startup. The practical consequence: **if PyTorch is ever
removed from this environment, layout detection quietly drops to CPU**
unless a system CUDA is installed.

Check which provider is actually in use — it is logged on every start:

```bash
grep "layout model on" run/logs/parse.log        # e.g. "layout model on CUDAExecutionProvider"
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

A `WARNING … not the GPU` line means the CUDA provider is installed but
could not load; install a system CUDA 12.x + cuDNN 9.x, or keep the
`nvidia-*-cu12` wheels in the environment.
- **Embedding** — `BAAI/bge-base-en-v1.5` (~440MB) downloads itself from
  Hugging Face into `~/.cache/huggingface/hub/` on the first embedding-service
  start; it is loaded from there on every start, so keep the cache.
- **Answering LLM** — see below.
- **Docling** (optional parser trial) downloads its models on first use of
  `scripts/docling_compare.py` or `PARSER_BACKEND = "docling"`.

`python scripts/hardware_check.py` prints what this machine's GPU/RAM can
run and the matching model choices.

## Answering model

**Local (default)** — Qwen3-4B via Ollama, natively installed:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b-instruct
```

Use the *instruct* (non-thinking) variant: the hybrid `qwen3:4b` ignores
Ollama's `think=false` and pollutes answers with chain-of-thought.

**Any OpenAI-compatible API** (a self-hosted gateway, a hosted model) —
open the UI's **Settings** page, choose backend `openai`, and enter base
URL, API key and model. Retrieved excerpts go to whichever model is
configured; streaming falls back to a whole answer if the API refuses it.

Settings are stored in `data/settings.json` (git-ignored). Environment
variables seed the defaults for scripting: `LLM_BACKEND`, `OLLAMA_MODEL`,
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and `EMBED_DEVICE`
(`cuda` for overnight batch embedding, `cpu` during the day so the LLM gets
the whole GPU).

## Web UI

The UI is a React app in `UI/frontend/`, built into `UI/static/` and served
by the UI service. Build it once (and after pulling frontend changes):

```bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"   # if node isn't on PATH
cd UI/frontend && npm install && npm run build
```

Without a build, the UI service falls back to the legacy single-page
`UI/index.html` (chat only). For frontend development: `npm run dev`
proxies `/v1` to the running UI service.

## Running

```bash
scripts/fresh_start.sh --yes         # once: purge + register every PDF + start the pipeline
scripts/start_query.sh               # daily: registry + embedder (CPU) + UI  → http://127.0.0.1:4002
scripts/start_query.sh --with-ingest # daily, but also parse + prune with the embedder on the GPU
scripts/stop_services.sh             # stop everything (sweeps orphans)
python scripts/pipeline_status.py    # progress snapshot
```

Details: [FRESH_START.md](FRESH_START.md), [DAILY_USE.md](DAILY_USE.md).

### Getting papers in

```bash
python scripts/register_pdfs.py                       # PDFs you copied into data/raw_pdfs/
cd download_manager && python downloader.py --once    # one arXiv crawl over config DOMAINS (capped per domain)
python downloader.py --once --domain "Graph Neural Networks" --max 5
python downloader.py --url https://arxiv.org/abs/2401.01234 --url https://host/paper.pdf
```

### Daily automatic ingestion (systemd user timer)

Easiest from the app: **Ingestion → Daily downloads**. Add topics with a cap
each, pick a time, flip the switch. That installs and enables the timer for
you. Only `loginctl enable-linger "$USER"` still needs a terminal, because it
needs root.

By hand:

```bash
ops/systemd/install.sh --max 20 --time 03:00        # install units (not enabled)
systemctl --user enable --now rag-daily-ingest.timer
loginctl enable-linger "$USER"                       # run while logged out too
```

Topics configured in the app live in `data/settings.json` under
`ingestion.schedule.topics`, and `ops.py daily-ingest` searches each one
separately so every topic carries its own cap. With no topics configured it
falls back to `download_manager/config.py`'s `DOMAINS` with a single shared
cap. Changing topics needs no reinstall; only the time of day is baked into
the unit file.

**Switching the timer on runs a cycle immediately.** `Persistent=true` treats
"never run" as a missed run — the same rule that makes a run missed overnight
fire five minutes after the next boot.

The job (`ops.py daily-ingest`) downloads one capped cycle, processes
everything pending, and stops what it started. A run missed because the
machine was off fires 5 minutes after the next boot.

### Parsing a single PDF by hand

```bash
python parse_manager/pdf_parser.py SomePaper.pdf      # looked up in data/raw_pdfs/ → data/parsed/SomePaper.json
python parse_manager/txt_processor.py SomePaper       # looked up in data/parsed/  → data/processed/SomePaper.txt
```

### Inspecting quality stage by stage

```bash
python scripts/rag_inspect.py parse    SomePaper.pdf --pages 3
python scripts/rag_inspect.py tables   SomePaper.pdf          # table crops + number check
python scripts/rag_inspect.py chunks   SomePaper
python scripts/rag_inspect.py retrieve "your question" --chats
python scripts/rag_inspect.py answer   "your question" --web
python scripts/docling_compare.py SomePaper.pdf --docling-native
```

See [USER_GUIDE.md](USER_GUIDE.md).

## Tests

```bash
python -m pytest tests/ -q
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: no such file: 'X.pdf'` | Path was relative to your shell's directory. Use a bare filename (auto-resolved to `data/raw_pdfs/`) or an absolute path. |
| `registry unreachable …` from a stage | Registry service (:4000) not running. One-off CLI runs still write files; the loops wait. `ops.py start-query` reuses a running registry or starts one. |
| A paper shows status `error` | The stage's exception is in `last_error` (Ingestion page or `pipeline_status.py`). Fix the cause, then `python scripts/register_pdfs.py --retry-errors`. |
| `CUDA out of memory` / GPU inference failed during parsing | Something else holds the GPU (Qwen, a training job). Parsing continues on CPU, slower. |
| Parsing is ~2 pages/s instead of 10–14 | The layout model is on CPU. `run/logs/parse.log` says which provider is in use, and warns explicitly when CUDA is installed but unusable. See "GPU for layout detection" below. |
| `has no ONNX export` on startup | New `.pt` weights were copied in. Run `python scripts/export_onnx.py` (needs ultralytics temporarily). |
| `port 4000 is already in use` | A stale service holds the port. `scripts/stop_services.sh` (= `ops.py stop --all`) sweeps orphans by pid, name and port. |
| UI shows the old single-page chat | `UI/static/` is missing — build the React app (above). |
| Ollama dot amber | Daemon up, configured model not pulled — `ollama pull <model>`. |
| Answers stream slowly | `ollama ps` shows a CPU/GPU split: the model loaded while the GPU was busy. `ollama stop <model>`; the start scripts do this automatically. |
| Retrieval returns nothing | Nothing embedded yet — run the ingestion pipeline. |
| Changed chunking / embedding model | Collection names are tied to the model (`embedding_manager/config.py`); run `scripts/fresh_start.sh --yes`. |
