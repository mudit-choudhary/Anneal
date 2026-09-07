# Setup & Running

## Requirements

- Linux, Python 3.12
- NVIDIA GPU with ≥4GB VRAM (GTX 1650 works; everything falls back to CPU)
- CUDA-enabled PyTorch

## Ports & services at a glance

| Service | Port | Started with | Needed for |
|---|---|---|---|
| registry_manager | 4000 | `cd registry_manager && python main.py` | status tracking (all pipeline services) |
| embedding_manager | 4001 | `cd embedding_manager && python main.py` | embedding + retrieval (stages 3–4, UI) |
| UI | 4002 | `cd UI && python main.py` | web querying |
| Ollama daemon | 11434 | systemd (automatic after install) | local LLM answers |

Parsing (stages 1–2) and the `rag_inspect.py` tool's `parse`/`chunks`
commands need **no services at all** — a missing registry only prints a
warning; files are still written.

## Environments

Each service historically has its own venv under `virtual_environments/`;
`globalragsetup_env` has the full dependency set and can run everything:

```bash
source virtual_environments/globalragsetup_env/bin/activate
```

To build a fresh env for one service:

```bash
python -m venv virtual_environments/<name>_env
virtual_environments/<name>_env/bin/pip install -r <service>/requirements.txt
```

## Models

The fine-tuned layout models live under `models/` (git-ignored — copy them in
from your training drive); the parser prefers
`yolo11s_doc_layout_imgsz_1024/weights/best.pt` and falls back to the nano
variant. To fetch the pretrained fallback instead:

```bash
python scripts/download_layout_model.py n   # n | s | m
```

## Local LLM (default query backend)

Answers are generated locally with Qwen3-4B via Ollama (runs natively, no
Docker needed — GPU passthrough via NVIDIA Container Toolkit would add setup
friction for zero benefit on a single machine):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b-instruct
```

Use the **instruct** (non-thinking) variant: the hybrid `qwen3:4b` ignores
Ollama's `think=false` flag and pollutes answers with chain-of-thought.

The Ollama daemon loads the model on the first query and frees VRAM after 30
minutes idle (`OLLAMA_KEEP_ALIVE` in `rag_setup/config.py`), so overnight
YOLO/embedding jobs get the GPU back automatically.

Backend selection and daytime VRAM:

```bash
export LLM_BACKEND=local            # default; or "gemini"
export GEMINI_API_KEY=...           # only for LLM_BACKEND=gemini
export EMBED_DEVICE=cpu             # query-time embedding on CPU keeps the
                                    # full 4GB free for the LLM (use cuda —
                                    # the default — for overnight batch runs)
```

## Running the pipeline

Start services in separate terminals (registry first):

```bash
# 1. Registry (port 4000) — required by all other services
cd registry_manager && python main.py

# 2. Downloader — fetches arXiv PDFs into data/raw_pdfs/
cd download_manager && python downloader.py

# 3. Parse manager — PDFs → layout JSON → tagged text
cd parse_manager && python main.py

# 4. Embedding manager (port 4001) — chunks + embeds into ChromaDB
cd embedding_manager && python main.py

# 5. Prune manager (optional) — deletes files that advanced past their stage
cd prune_manager && python pruning.py

# Query — terminal CLI…
cd rag_setup && python rag.py

# …or web UI at http://127.0.0.1:4002
cd UI && python main.py
```

### Parsing a single PDF by hand

Bare filenames resolve automatically against the data directories, so these
work from any current directory:

```bash
python parse_manager/pdf_parser.py SomePaper.pdf      # looked up in data/raw_pdfs/ → data/parsed/SomePaper.json
python parse_manager/txt_processor.py SomePaper       # looked up in data/parsed/  → data/processed/SomePaper.txt
```

(The registry being down just prints a warning; files are still written.)

### Inspecting quality stage by stage

`scripts/rag_inspect.py` inspects each stage of the RAG cycle (parsing,
chunking, retrieval, answers) — see [USER_GUIDE.md](USER_GUIDE.md) for the
full tuning workflow:

```bash
python scripts/rag_inspect.py parse    SomePaper.pdf --pages 3
python scripts/rag_inspect.py chunks   SomePaper
python scripts/rag_inspect.py retrieve "your question"
python scripts/rag_inspect.py answer   "your question"
```

## Tests

```bash
python -m pytest tests/ -q
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: no such file: 'X.pdf'` | Path was relative to your shell's current directory. Use a bare filename (auto-resolved to `data/raw_pdfs/`) or an absolute path. |
| `[parser] registry unreachable … status not updated` | Registry service (:4000) not running. Harmless for one-off runs; required for the automated pipeline loops. |
| `CUDA out of memory; falling back to CPU` during parsing | Something else holds the GPU (a training job, or Qwen still loaded — check `nvidia-smi` / `ollama ps`; `ollama stop qwen3:4b-instruct` frees it). Parsing continues on CPU, just slower. |
| `ModuleNotFoundError` in a service | Wrong venv active — `globalragsetup_env` has everything; per-service envs may be stale. |
| UI dot for Ollama is amber | Daemon up but `qwen3:4b-instruct` not pulled — `ollama pull qwen3:4b-instruct`. |
| Answers contain reasoning rambling | You're on the hybrid `qwen3:4b` model; use `qwen3:4b-instruct` (see Local LLM above). |
| Retrieval returns nothing | Nothing embedded yet, or the embedding service points at an empty `vector_db/` — run the ingestion pipeline first. |

Heuristics tests run in milliseconds without a GPU. The end-to-end test needs
the layout model and a PDF in `data/raw_pdfs/` (2 pages, CPU-safe), and skips
itself otherwise.
