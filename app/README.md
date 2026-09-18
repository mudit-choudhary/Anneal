# Anneal — Local-First RAG & Layout-Aware Chunking Pipeline

The application half of [Anneal](../README.md): six services that turn research
PDFs into answers you can trace back to a page, on one 4 GB consumer GPU with no
cloud dependency in the parse or embed path. The evaluation harness that judges
it lives in [evals/](../evals/README.md).

A paper moves through four stages, each one recorded in the registry so any
stage can be resumed or re-run:

```
download ──▶ parse ──▶ embed ──▶ answer
  arXiv      YOLOv11 layout    bge-base    retrieval + local LLM
  or a URL   + assembler       in Chroma   with citations
```

## Quick start

From the repository root:

```bash
python3.12 -m venv virtual_environments/annealenv                      # once
virtual_environments/annealenv/bin/pip install -r app/requirements.txt
virtual_environments/annealenv/bin/pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
ln -s "$PWD/app/bin/anneal" ~/.local/bin/anneal                        # once
anneal                                        # start everything, open the app
```

That last install line is not optional: `chromadb` pulls in the CPU
`onnxruntime`, which shares a package directory with the GPU build and silently
replaces it. The note at the top of [requirements.txt](requirements.txt)
explains it.

## Commands

| Command | Does |
|---|---|
| `anneal` | start everything and open <http://127.0.0.1:4002> (`--no-open` to skip) |
| `anneal status` | what is running, what is indexed |
| `anneal stop [--all]` | stop everything (`--all` also sweeps orphans) |
| `anneal --with-ingest` | also run parse and prune, embedder on the GPU |
| `anneal --port 8080` | move the web UI (also `--registry-port`, `--embedding-port`) |
| `python scripts/ops.py fresh-start --yes` | purge and re-ingest every PDF |
| `python scripts/ops.py daily-ingest --max 20` | one download cycle, process, stop |

## Services

| Service | Port | Does |
|---|---|---|
| `registry_manager` | 4000 | one row per paper and its stage; everything else needs it |
| `parse_manager` | — | PDF → YOLO layout → assembled, tagged text |
| `embedding_manager` | 4001 | chunking, embedding, retrieval (papers and saved chats) |
| `prune_manager` | — | clears intermediates once embedded; raw PDFs kept or archived |
| `download_manager` | — | arXiv by topic or a pasted link |
| `UI` | 4002 | React app + JSON API, streaming answers with citations |

Ports come from `ANNEAL_UI_PORT`, `ANNEAL_REGISTRY_PORT` and
`ANNEAL_EMBEDDING_PORT`, which the `anneal --port …` flags set.

## Where your data lives

Nothing the app writes is kept in the repository — papers, models, the vector
store, databases, logs and pids all go to `~/.local/share/anneal/` (or
`$ANNEAL_HOME`). [common/paths.py](common/paths.py) is the only place that
resolves it; [docs/OPERATIONS.md](docs/OPERATIONS.md) has the layout and how to
move it.

## Layout

```
bin/anneal          the one command
common/             paths, logging, settings, chat store, registry client
download_manager/   parse_manager/      prune_manager/
embedding_manager/  registry_manager/   five of the six services
UI/                 the sixth: main.py (API) · frontend/ (React) · static/ (build)
rag_setup/          retrieval + prompt + answer, shared by the UI and the CLI
scripts/            ops.py and the inspection tools
ops/systemd/        daily-ingest timer units + install.sh
docs/  diagrams/
```

> `npm run dev` serves the frontend **only**, with no backend behind it. Use
> `anneal`.

## Documentation

- [docs/OPERATIONS.md](docs/OPERATIONS.md) — install, run, ingest, rebuild, troubleshoot
- [docs/PARSING.md](docs/PARSING.md) — layout detection, the assembler, fine-tuning the model
- [docs/SYSTEM_WALKTHROUGH.md](docs/SYSTEM_WALKTHROUGH.md) — every module and store: what, why, when, what follows
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — testing and tuning each RAG stage
- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) — how the system got here: decisions, benchmarks, what was rejected
- [docs/PENDING_IMPROVEMENTS.md](docs/PENDING_IMPROVEMENTS.md) — known gaps, deliberately deferred
- [diagrams/system_architecture_detailed.drawio](diagrams/system_architecture_detailed.drawio) — multi-page: system + one page per module
