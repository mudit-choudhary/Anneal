#!/usr/bin/env bash
# Round 2, start or resume. Every stage skips work already saved, so after a
# crash or reboot, running this again continues where it stopped.
#
#     bash evals/scripts/round2_resume.sh 2>&1 | tee -a evals/Reports/round2_run.log
#
# Needs Ollama running (qwen3:4b-instruct for answers, gemma3:4b for judging).
set -euo pipefail
cd "$(dirname "$0")/../.."
source virtual_environments/globalragsetup_env/bin/activate

echo "== $(date '+%F %T') parse: cached papers are skipped"
python -u evals/scripts/parse_cache.py

# Once the retrieval run has answered anything, the question set is frozen:
# every cell must answer the same 400 questions.
if [ -n "$(ls -A evals/Reports/rag_round2_rows 2>/dev/null)" ]; then
    echo "== questions: frozen, retrieval run already started"
else
    echo "== $(date '+%F %T') questions: papers with questions are skipped"
    source evals/keys_export.sh
    python -u evals/scripts/build_questions.py --shortlist 400 --workers 3
fi

echo "== $(date '+%F %T') retrieval: finished cells and answered questions are skipped"
python -u evals/scripts/run_rag_eval.py --device cuda

echo "== $(date '+%F %T') done"
