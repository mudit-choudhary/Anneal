#!/usr/bin/env bash
# Thin wrapper: see scripts/ops.py fresh-start
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/virtual_environments/globalragsetup_env/bin/python" "$ROOT/scripts/ops.py" fresh-start "$@"
