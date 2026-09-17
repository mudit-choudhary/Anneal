# Research-Paper RAG Setup

A local RAG system over research papers, built to run on a single machine
with a 4GB GPU (GTX 1650). Papers are downloaded from arXiv (or by link),
parsed with a **fine-tuned YOLOv11 document-layout model** (paragraph-accurate,
layout-aware extraction), chunked structure-aware and embedded with bge-base
into ChromaDB, and queried through a **React web app** backed by a locally
hosted Qwen3-4B (Ollama) or any OpenAI-compatible API — with citations,
optional web search, saved conversations as memory, and Mermaid rendering.

## Results

The parsing and chunking choices here were evaluated twice, the second time with
the metrics, tests and thresholds fixed in writing **before** the run:

**→ [evals/Reports/Report.md](evals/Reports/Report.md)** — 3 parsers x 3 chunkers,
400 questions, 514 papers, every comparison paired and corrected for multiplicity.

- The fine-tuned YOLOv11 parser **retrieves significantly better** than Docling and
  PyMuPDF4LLM under every chunker tested, and loses the fewest answers in parsing
  (24 of 400, against 48 and 61).
- Structure-aware chunking beats a fixed-token window; against a recursive character
  splitter the difference is **not established**, and the report says so rather than
  moving the threshold.
- Faithfulness and context sufficiency **could not be measured reliably** by either
  method tried; both were demoted before any comparison was run.

## Documentation

- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) — how the system got here: decisions, benchmarks, what was rejected
- [docs/DAILY_USE.md](docs/DAILY_USE.md) — morning start, asking questions, adding papers
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — testing & tuning each RAG stage
- [docs/FRESH_START.md](docs/FRESH_START.md) — purge everything and re-ingest
- [docs/PENDING_IMPROVEMENTS.md](docs/PENDING_IMPROVEMENTS.md) — known gaps, deliberately deferred
- [docs/architecture.md](docs/architecture.md) — services, data flow, file lifecycle (short)
- [docs/SYSTEM_WALKTHROUGH.md](docs/SYSTEM_WALKTHROUGH.md) — every module and store: what, why, when, what follows
- [diagrams/system_architecture_detailed.drawio](diagrams/system_architecture_detailed.drawio) — multi-page: system + one page per module
- [docs/parse_manager.md](docs/parse_manager.md) — layout-aware parsing design
- [docs/AssemblerLogic.md](docs/AssemblerLogic.md) — the assembler: input format, reading order, paragraph reconstruction, output
- [evals/README.md](evals/README.md) — the parser x chunker harness and its report
- [docs/setup.md](docs/setup.md) — environments, models, building the UI, running, scheduling
- [docs/yolo_finetuning.md](docs/yolo_finetuning.md) — fine-tuning the layout model

## Layout

```
common/  docs/  diagrams/  tests/  UI/  ops/  scripts/
download_manager/  parse_manager/  prune_manager/
embedding_manager/  registry_manager/  rag_setup/
models/*  data/*  vector_db/*  run/*              (* git-ignored)
```

## Quick start

```bash
cd UI/frontend && npm install && npm run build && cd ../..   # once: build the web app
scripts/fresh_start.sh --yes      # once: purge + ingest every PDF in data/raw_pdfs/
scripts/start_query.sh            # daily: start query services → http://127.0.0.1:4002
scripts/stop_services.sh          # stop everything
```

Tests: `python -m pytest tests/ -q`
