#!/usr/bin/env bash
# Fresh start: purge every piece of pipeline state and re-ingest all PDFs in
# data/raw_pdfs/ from scratch. Raw PDFs are never touched.
#
#   scripts/fresh_start.sh              # shows what will be deleted, asks to confirm
#   scripts/fresh_start.sh --yes        # no prompt
#   scripts/fresh_start.sh --limit 3    # register only the first 3 PDFs (smoke test)
#   scripts/fresh_start.sh --no-ui      # don't start the web UI
#
# Steps: stop services -> delete vector_db/, registry DB, data/parsed/,
# data/processed/ -> start registry -> register PDFs -> start parse_manager,
# embedding_manager, prune_manager (+ UI). Services run in the background
# with logs in run/logs/ and PIDs in run/pids/.
#
#   python scripts/pipeline_status.py     # watch progress
#   python scripts/wait_for_ingestion.py  # block until done (chain a shutdown)
#   scripts/stop_services.sh              # stop everything
#
# For day-to-day querying (no purge), use scripts/start_query.sh instead.
# The downloader is NOT started here; run it separately when you want new
# papers:  cd download_manager && python downloader.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/service_lib.sh"

YES=0; LIMIT=""; UI=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES=1 ;;
    --limit) LIMIT="$2"; shift ;;
    --no-ui) UI=0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
  shift
done

echo "== 1/5 stopping any running services"
"$ROOT/scripts/stop_services.sh"
unload_ollama_models   # free the GPU for YOLO + the embedder

echo "== 2/5 purge"
"$PY" "$ROOT/scripts/reset_ingestion.py"
if [[ $YES -ne 1 ]]; then
  read -r -p "Delete all of the above? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || { echo "aborted"; exit 1; }
fi
"$PY" "$ROOT/scripts/reset_ingestion.py" --yes >/dev/null

echo "== 3/5 registry"
for p in 4000 4001 4002; do require_port_free "$p"; done
start registry registry_manager main.py
wait_for_http registry http://127.0.0.1:4000/openapi.json 30

echo "== 4/5 registering PDFs"
"$PY" "$ROOT/scripts/register_pdfs.py" ${LIMIT:+--limit "$LIMIT"}

echo "== 5/5 pipeline services"
start parse parse_manager main.py
start embedding embedding_manager main.py EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
start prune prune_manager pruning.py
if [[ $UI -eq 1 ]]; then
  start ui UI main.py
fi

cat <<EOF

Ingestion running. The embedding model downloads on first start (~440MB).
  progress : python scripts/pipeline_status.py
  wait     : python scripts/wait_for_ingestion.py   (&& scripts/stop_services.sh && systemctl poweroff)
  logs     : tail -f run/logs/*.log
  stop     : scripts/stop_services.sh
EOF
[[ $UI -eq 1 ]] && echo "  UI       : http://127.0.0.1:4002"
exit 0
