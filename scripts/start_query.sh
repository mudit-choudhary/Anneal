#!/usr/bin/env bash
# Daily start: bring up just what asking questions needs — registry,
# embedding service (on CPU, so the whole GPU stays free for Qwen), and the
# web UI. Nothing is purged; the vector store and registry are reused as-is.
#
#   scripts/start_query.sh                 # query only
#   scripts/start_query.sh --with-ingest   # also parse + prune, embedder on GPU
#                                          # (use when adding papers today)
#   scripts/stop_services.sh               # end of day (optional — idle
#                                          # services cost nothing; Ollama
#                                          # unloads Qwen by itself)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/service_lib.sh"

INGEST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ingest) INGEST=1 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
  shift
done

echo "== stopping any previous services"
"$ROOT/scripts/stop_services.sh"
for p in 4000 4001 4002; do require_port_free "$p"; done

echo "== starting"
check_ollama
unload_ollama_models
start registry registry_manager main.py
wait_for_http registry http://127.0.0.1:4000/openapi.json 30

if [[ $INGEST -eq 1 ]]; then
  start embedding embedding_manager main.py EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
  start parse parse_manager main.py
  start prune prune_manager pruning.py
else
  start embedding embedding_manager main.py EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
fi
start ui UI main.py

echo "== waiting for the embedding model to load (CPU: ~20s)"
wait_for_http embedding http://127.0.0.1:4001/list_files 120
wait_for_http ui http://127.0.0.1:4002/api/status 30

papers=$(curl -sf http://127.0.0.1:4001/list_files | "$PY" -c "import sys,json; print(len(json.load(sys.stdin)['files']))" 2>/dev/null || echo "?")
cat <<EOF

Ready — $papers papers in the vector store.
  UI   : http://127.0.0.1:4002   (first question loads Qwen: ~10-20s)
  CLI  : cd rag_setup && python rag.py
  stop : scripts/stop_services.sh
EOF
[[ $INGEST -eq 1 ]] && echo "  add papers: copy PDFs to data/raw_pdfs/ then: python scripts/register_pdfs.py"
exit 0
