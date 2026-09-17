#!/usr/bin/env bash
# Thin wrapper kept for compatibility; `anneal` (app/bin/anneal) is the same code.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/bin/anneal" stop --all "$@"
