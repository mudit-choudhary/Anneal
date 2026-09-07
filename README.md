# Research-Paper RAG Setup

A local RAG system over research papers, built to run on a single machine with
a 4GB GPU (GTX 1650). Papers are downloaded from arXiv, parsed with a
**fine-tuned YOLOv11 document-layout model** (paragraph-accurate, layout-aware
extraction), chunked and embedded into ChromaDB, and queried through a CLI
backed by a **locally hosted Qwen3-4B (Ollama)** by default, with Gemini as an
optional fallback.

## Documentation

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — testing & tuning each RAG stage
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
source virtual_environments/globalragsetup_env/bin/activate
cd registry_manager && python main.py          # registry :4000
cd parse_manager && python main.py             # layout-aware parsing
cd embedding_manager && python main.py         # embeddings :4001
cd rag_setup && python rag.py                  # query CLI (local Qwen3-4B via Ollama)
cd UI && python main.py                        # or web UI → http://127.0.0.1:4002
```

Tests: `python -m pytest tests/ -q`
