#!/usr/bin/env bash
# Stop every service started by scripts/fresh_start.sh.
#
# Kills each recorded PID together with its descendants (uvicorn can leave a
# child holding the port), then sweeps the service ports for orphans from
# earlier runs so a stale process can never shadow a fresh one.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS="$ROOT/run/pids"
PORTS="4000 4001 4002"

kill_tree() {  # TERM a process and everything below it, escalating to KILL
  local pid=$1 sig=${2:-TERM} child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child" "$sig"; done
  kill "-$sig" "$pid" 2>/dev/null
}

stopped=0
if [[ -d "$PIDS" ]]; then
  for pidfile in "$PIDS"/*.pid; do
    [[ -e "$pidfile" ]] || continue
    name=$(basename "$pidfile" .pid)
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill_tree "$pid"
      for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill_tree "$pid" KILL
      echo "  stopped $name (pid $pid)"
      stopped=$((stopped + 1))
    fi
    rm -f "$pidfile"
  done
fi

# Orphan sweep 1: any pipeline service process from this repo's venv that
# wasn't in a pid file (loops like parse/prune hold no port, so this is the
# only way to find them).
for pid in $(pgrep -f "$ROOT/virtual_environments/.*/python (main\.py|pruning\.py|downloader\.py)$" 2>/dev/null); do
  [[ -r "/proc/$pid/cmdline" ]] || continue
  what=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | awk '{print $NF}')
  kill_tree "$pid"
  sleep 1
  kill -0 "$pid" 2>/dev/null && kill_tree "$pid" KILL
  echo "  stopped orphan service (pid $pid: $what)"
  stopped=$((stopped + 1))
done

# Orphan sweep 2: anything else still listening on a service port.
for port in $PORTS; do
  for pid in $(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
    [[ -r "/proc/$pid/cmdline" ]] || continue   # already gone
    if tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "$ROOT"; then
      kill_tree "$pid"
      sleep 1
      kill -0 "$pid" 2>/dev/null && kill_tree "$pid" KILL
      echo "  stopped orphan on :$port (pid $pid)"
      stopped=$((stopped + 1))
    else
      echo "  WARNING: port $port is held by pid $pid, which is not one of ours — stop it manually"
    fi
  done
done

[[ $stopped -eq 0 ]] && echo "  nothing was running"
exit 0
