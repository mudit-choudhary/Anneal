#!/usr/bin/env bash
# Install the daily-ingest timer as a *user* systemd unit (no sudo).
#
#   ops/systemd/install.sh [--max N] [--time HH:MM]     # default: 20 papers/domain at 03:00
#   systemctl --user enable --now rag-daily-ingest.timer # then enable it
#   systemctl --user list-timers                         # verify
#   journalctl --user -u rag-daily-ingest -f              # logs (also run/logs/daily-ingest.log)
#
# User timers only run while you are logged in unless lingering is enabled:
#   loginctl enable-linger "$USER"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/virtual_environments/globalragsetup_env/bin/python"
MAX=20; TIME="03:00"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max) MAX="$2"; shift ;;
    --time) TIME="$2"; shift ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
  shift
done

UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR" "$ROOT/run/logs"
for f in rag-daily-ingest.service rag-daily-ingest.timer; do
  sed -e "s#__ROOT__#$ROOT#g" -e "s#__PY__#$PY#g" -e "s#__MAX__#$MAX#g" -e "s#__TIME__#$TIME#g" \
      "$ROOT/ops/systemd/$f" > "$UNIT_DIR/$f"
  echo "installed $UNIT_DIR/$f"
done
systemctl --user daemon-reload
cat <<EOF

Installed (not enabled). To activate:
  systemctl --user enable --now rag-daily-ingest.timer
  loginctl enable-linger $USER      # run even when you're not logged in
Runs daily at $TIME, up to $MAX new papers per domain; a missed run fires 5 min after boot.
EOF
