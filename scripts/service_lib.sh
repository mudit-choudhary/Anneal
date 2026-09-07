#!/usr/bin/env bash
# Shared helpers for starting pipeline services in the background.
# Sourced by fresh_start.sh and start_query.sh — not run directly.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/virtual_environments/globalragsetup_env/bin/python"
LOGS="$ROOT/run/logs"
PIDS="$ROOT/run/pids"

[[ -x "$PY" ]] || { echo "venv python not found at $PY"; exit 1; }
mkdir -p "$LOGS" "$PIDS"

require_port_free() {  # a stale process on a service port would shadow the new one
  local port=$1 pid
  pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
  if [[ -n "$pid" ]]; then
    echo "port $port is already in use (pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-80))"
    echo "stop it first (scripts/stop_services.sh, or kill $pid), then re-run."
    exit 1
  fi
}

start() {  # start <name> <dir> <script> [ENV=VAL ...]
  local name=$1 dir=$2 script=$3; shift 3
  # No subshell around the '&': backgrounding a `( cd && cmd )` list makes
  # bash fork a wrapper whose pid would be recorded while the real Python
  # process is its child — which is exactly the orphan stop_services.sh
  # then can't find. nohup -> env -> python all exec in place, so $! is the
  # service itself.
  pushd "$ROOT/$dir" >/dev/null
  nohup env PYTHONUNBUFFERED=1 "$@" "$PY" "$script" >"$LOGS/$name.log" 2>&1 </dev/null &
  echo $! >"$PIDS/$name.pid"
  popd >/dev/null
  echo "  started $name (pid $(cat "$PIDS/$name.pid"), log run/logs/$name.log)"
}

wait_for_http() {  # wait_for_http <name> <url> [seconds]
  local name=$1 url=$2 secs=${3:-30}
  for _ in $(seq 1 "$secs"); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    kill -0 "$(cat "$PIDS/$name.pid" 2>/dev/null)" 2>/dev/null || break
    sleep 1
  done
  echo "$name did not come up; see run/logs/$name.log"
  return 1
}

check_ollama() {
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "  ollama daemon: running"
  else
    echo "  ollama daemon: NOT running — local answers will fail. Start it with: sudo systemctl start ollama"
  fi
}

unload_ollama_models() {
  # Ollama decides a model's CPU/GPU split when it loads and keeps it while
  # the model stays warm. A model loaded while the GPU embedder was resident
  # sits ~70% on CPU (slow) until unloaded — so start each session clean;
  # the next question reloads it onto whatever GPU memory is free (~10s).
  local m
  for m in $(ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do
    ollama stop "$m" >/dev/null 2>&1 && echo "  unloaded $m from ollama (reloads on first question)"
  done
}
