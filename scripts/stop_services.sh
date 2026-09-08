#!/usr/bin/env bash
# Thin wrapper: see scripts/ops.py stop  (--all sweeps orphans by name and port)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/virtual_environments/globalragsetup_env/bin/python" "$ROOT/scripts/ops.py" stop --all "$@"
