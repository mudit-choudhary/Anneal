# Research-Paper RAG Setup

A local RAG system over research papers, built to run on a single machine with
a 4GB GPU (GTX 1650). Papers are downloaded from arXiv, parsed with a
**fine-tuned YOLOv11 document-layout model** (paragraph-accurate, layout-aware
extraction), chunked structure-aware and embedded with bge-base into ChromaDB,
and queried through a CLI
backed by a **locally hosted Qwen3-4B (Ollama)** by default, with Gemini as an
optional fallback.

## Documentation

- [docs/DAILY_USE.md](docs/DAILY_USE.md) — morning start, asking questions, adding papers
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — testing & tuning each RAG stage
- [docs/FRESH_START.md](docs/FRESH_START.md) — purge everything and re-ingest (`scripts/fresh_start.sh`)
- [docs/PENDING_IMPROVEMENTS.md](docs/PENDING_IMPROVEMENTS.md) — known gaps, deliberately deferred
- [docs/architecture.md](docs/architecture.md) — services, data flow, file lifecycle
- [docs/parse_manager.md](docs/parse_manager.md) — layout-aware parsing design
- [docs/setup.md](docs/setup.md) — environments, models, running the pipeline
- [docs/yolo_finetuning.md](docs/yolo_finetuning.md) — fine-tuning the layout model
- [diagrams/](diagrams/) — drawio architecture diagrams

## Layout

```
docs/  diagrams/  tests/  UI/
download_manager/  parse_manager/  prune_manager/
embedding_manager/  registry_manager/  rag_setup/
scripts/  models/*  data/*  vector_db/*        (* git-ignored)
```

## Quick start

```bash
scripts/fresh_start.sh --yes      # once: purge + ingest every PDF in data/raw_pdfs/
scripts/start_query.sh            # daily: start query services → http://127.0.0.1:4002
scripts/stop_services.sh          # stop everything
```

Tests: `python -m pytest tests/ -q`
