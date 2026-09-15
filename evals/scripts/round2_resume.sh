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

# Once the pre-registration records a dataset hash, the question set is frozen:
# every cell must answer the same 400 questions, so it is never rebuilt, and the
# run refuses to start if dataset.json no longer matches the recorded hash.
FROZEN_SHA=$(python -c "import json; print(json.load(open('evals/Reports/round2_preregistration.json')).get('dataset_sha256') or '')")
if [ -n "$FROZEN_SHA" ]; then
    ACTUAL_SHA=$(sha256sum evals/questions/dataset.json | cut -d' ' -f1)
    if [ "$ACTUAL_SHA" != "$FROZEN_SHA" ]; then
        echo "!! dataset.json does not match the frozen hash; refusing to run" >&2
        exit 1
    fi
    echo "== questions: frozen, hash verified"
else
    echo "== $(date '+%F %T') questions: papers with questions are skipped"
    source evals/keys_export.sh
    python -u evals/scripts/build_questions.py --shortlist 400 --workers 3
fi

echo "== $(date '+%F %T') retrieval: finished cells and answered questions are skipped"
python -u evals/scripts/run_rag_eval.py --device cuda

echo "== $(date '+%F %T') done"
