"""Cross-platform process management for the pipeline (replaces the bash
launchers; the .sh files are thin wrappers around this).

    anneal                                                           start everything, open the UI
    anneal status                                                    what is running, what is indexed
    anneal stop [--all]                                              stop everything
    python scripts/ops.py fresh-start [--yes] [--limit N] [--no-ui]   purge + register + start all
    python scripts/ops.py start-query [--with-ingest]                 daily start (registry, embedder on CPU, UI)
    python scripts/ops.py stop [--all]                                stop services (+ orphan sweep with --all)
    python scripts/ops.py daily-ingest [--max N]                      download one cycle, process until done, stop
    python scripts/ops.py start <service>... / restart <service>...

Services run detached; logs in run/logs/<name>.log, pids in run/pids/. A
registry that is already healthy (e.g. started elsewhere) is reused rather
than duplicated.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import (  # noqa: E402
    EMBEDDING_URL, LOG_DIR, PID_DIR, REGISTRY_URL, APP_ROOT, SERVICE_PORTS, UI_URL, VENV_PYTHON,
)
from common.registry_client import RegistryClient  # noqa: E402

SERVICES = {
    "registry": ("registry_manager", "main.py"),
    "parse": ("parse_manager", "main.py"),
    "embedding": ("embedding_manager", "main.py"),
    "prune": ("prune_manager", "pruning.py"),
    "download": ("download_manager", "downloader.py"),
    "ui": ("UI", "main.py"),
}

# Shown in the UI's control pane so each switch explains itself.
SERVICE_INFO = {
    "registry": ("Registry", "Tracks every paper's stage. Everything else needs it.", 4000),
    "parse": ("Parser", "PDF -> layout -> structured text. Needed to process new papers.", None),
    "embedding": ("Embedder", "Chunks and embeds; also answers searches.", 4001),
    "prune": ("Pruner", "Clears intermediates once a paper is embedded.", None),
    "download": ("Downloader", "Crawls arXiv on a schedule for new papers.", None),
    "ui": ("Web UI", "This page. Stop it from the terminal, not from itself.", 4002),
}
SERVICE_SCRIPTS = ("main.py", "pruning.py", "downloader.py")
IS_WINDOWS = os.name == "nt"


def say(msg):
    print(msg, flush=True)


# --------------------------------------------------------------- process helpers
def pid_of(name):
    try:
        return int((PID_DIR / f"{name}.pid").read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def running(name):
    pid = pid_of(name)
    return bool(pid and psutil.pid_exists(pid))


def ours(proc):
    try:
        cmd = " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return str(APP_ROOT) in cmd and any(cmd.endswith(s) or f" {s} " in cmd + " " for s in SERVICE_SCRIPTS)


def kill_tree(pid, grace=5):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = proc.children(recursive=True) + [proc]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def port_owner(port):
    for c in psutil.net_connections(kind="inet"):
        if c.status == psutil.CONN_LISTEN and c.laddr.port == port and c.pid:
            return c.pid
    return None


def http_ok(url, timeout=2):
    import requests
    try:
        return requests.get(url, timeout=timeout).ok
    except requests.RequestException:
        return False


def wait_http(url, secs, name):
    for _ in range(secs):
        if http_ok(url):
            return True
        if not running(name):
            break
        time.sleep(1)
    say(f"  {name} did not come up; see run/logs/{name}.log")
    return False


# --------------------------------------------------------------- start / stop
def start(name, env=None):
    subdir, script = SERVICES[name]
    if running(name):
        say(f"  {name} already running (pid {pid_of(name)})")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"{name}.log", "a", buffering=1)
    kwargs = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([str(VENV_PYTHON), script], cwd=APP_ROOT / subdir,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}, **kwargs)
    (PID_DIR / f"{name}.pid").write_text(str(proc.pid))
    say(f"  started {name} (pid {proc.pid}, log run/logs/{name}.log)")


def stop(names=None, sweep=False):
    stopped = 0
    for pidfile in sorted(PID_DIR.glob("*.pid")) if PID_DIR.exists() else []:
        name = pidfile.stem
        if names and name not in names:
            continue
        pid = pid_of(name)
        if pid and psutil.pid_exists(pid):
            kill_tree(pid)
            say(f"  stopped {name} (pid {pid})")
            stopped += 1
        pidfile.unlink(missing_ok=True)
    if sweep:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            if ours(proc):
                kill_tree(proc.pid)
                say(f"  stopped orphan service (pid {proc.pid})")
                stopped += 1
        for name, port in SERVICE_PORTS.items():
            pid = port_owner(port)
            if pid:
                try:
                    p = psutil.Process(pid)
                    if str(APP_ROOT) in " ".join(p.cmdline()):
                        kill_tree(pid)
                        say(f"  stopped orphan on :{port} (pid {pid})")
                        stopped += 1
                    else:
                        say(f"  WARNING: port {port} is held by pid {pid} ({p.name()}), not one of ours")
                except psutil.NoSuchProcess:
                    pass
    if not stopped:
        say("  nothing was running")


def require_ports_free(ports):
    for port in ports:
        pid = port_owner(port)
        if pid:
            say(f"port {port} is already in use (pid {pid}); run `anneal stop --all` or free it, then retry")
            sys.exit(1)


def ensure_registry():
    """Reuse a healthy registry (started elsewhere) or start one."""
    if RegistryClient().health():
        if not running("registry"):
            say("  registry already running (not started by ops.py) — reusing it")
        return True
    require_ports_free([SERVICE_PORTS["registry"]])
    start("registry")
    return wait_http(f"{REGISTRY_URL}/v1/health", 30, "registry")


def unload_ollama_models():
    """Ollama fixes a model's CPU/GPU split at load time; start each session
    clean so the next question loads onto whatever GPU memory is free."""
    if not shutil.which("ollama"):
        return
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return
    for line in out.splitlines()[1:]:
        model = line.split()[0] if line.split() else None
        if model:
            subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
            say(f"  unloaded {model} from ollama (reloads on first question)")


def check_ollama():
    say("  ollama daemon: " + ("running" if http_ok("http://127.0.0.1:11434/api/version") else
                               "NOT running — local answers will fail (sudo systemctl start ollama)"))


def ensure_ui_built():
    """The UI service serves UI/static; `npm run dev` bypasses it and serves the
    frontend with no backend behind it, which is the usual first-run confusion."""
    if (APP_ROOT / "UI" / "static" / "index.html").exists():
        return
    npm = shutil.which("npm")
    if not npm:
        sys.exit("the web UI is not built and npm was not found.\n"
                 "  install Node, then: cd UI/frontend && npm install && npm run build")
    say("== building the web UI (first run only, ~1 min)")
    front = APP_ROOT / "UI" / "frontend"
    subprocess.run([npm, "install"], cwd=front, check=True)
    subprocess.run([npm, "run", "build"], cwd=front, check=True)


# --------------------------------------------------------------- commands
def cmd_up(a):
    """Bare `anneal`: everything a person needs to start asking questions."""
    ensure_ui_built()
    cmd_start_query(a)
    if not a.no_open:
        import webbrowser
        if webbrowser.open(UI_URL):
            say(f"  opened {UI_URL}")


def cmd_status(a):
    import requests
    say(f"  {'service':<11}{'state':<11}{'port':<7}what it does")
    for name in SERVICES:
        label, desc, port = SERVICE_INFO[name]
        state = f"running" if running(name) else "-"
        say(f"  {label:<11}{state:<11}{str(port or ''):<7}{desc}")
    ollama = http_ok("http://127.0.0.1:11434/api/version")
    say(f"\n  ollama     {'running' if ollama else 'NOT running'}")
    say(f"  web UI     {UI_URL} {'reachable' if http_ok(f'{UI_URL}/v1/status') else 'not reachable'}")
    try:
        stats = RegistryClient().stats()
        say(f"  registry   {stats['total']} papers: " +
            ", ".join(f"{k} {v}" for k, v in sorted(stats["counts"].items())))
    except Exception:                                            # noqa: BLE001
        say("  registry   not reachable")
    try:
        n = len(requests.get(f"{EMBEDDING_URL}/v1/papers", timeout=3).json()["papers"])
        say(f"  vectors    {n} papers in the vector store")
    except Exception:                                            # noqa: BLE001
        say("  vectors    embedder not reachable")


def cmd_fresh_start(a):
    say("== 1/5 stopping any running services")
    stop(sweep=True)
    unload_ollama_models()
    say("== 2/5 purge")
    sys.path.insert(0, str(APP_ROOT / "scripts"))
    import reset_ingestion
    reset_ingestion.reset(apply=False)
    if not a.yes:
        if input("Delete all of the above? [y/N] ").strip().lower() != "y":
            say("aborted")
            sys.exit(1)
    reset_ingestion.reset(apply=True, quiet=True)
    say("== 3/5 registry")
    require_ports_free(SERVICE_PORTS.values())
    if not ensure_registry():
        sys.exit(1)
    say("== 4/5 registering PDFs")
    cmd = [str(VENV_PYTHON), str(APP_ROOT / "scripts" / "register_pdfs.py")] + (["--limit", str(a.limit)] if a.limit else [])
    subprocess.run(cmd, check=False)
    say("== 5/5 pipeline services")
    start("parse")
    start("embedding", {"EMBED_DEVICE": os.environ.get("EMBED_DEVICE", "cuda")})
    start("prune")
    if not a.no_ui:
        start("ui")
    say("\nIngestion running. The embedding model downloads on first start (~440MB).\n"
        "  progress : anneal status\n"
        "  wait     : python app/scripts/wait_for_ingestion.py   (&& anneal stop && systemctl poweroff)\n"
        "  stop     : anneal stop" + ("" if a.no_ui else f"\n  UI       : {UI_URL}"))


def cmd_start_query(a):
    say("== stopping any previous services")
    stop()
    say("== starting")
    check_ollama()
    unload_ollama_models()
    if not ensure_registry():
        sys.exit(1)
    require_ports_free([SERVICE_PORTS["embedding"], SERVICE_PORTS["ui"]])
    if a.with_ingest:
        start("embedding", {"EMBED_DEVICE": os.environ.get("EMBED_DEVICE", "cuda")})
        start("parse")
        start("prune")
    else:
        start("embedding", {"EMBED_DEVICE": os.environ.get("EMBED_DEVICE", "cpu")})
    start("ui")
    say("== waiting for the embedding model to load (CPU: ~20s)")
    wait_http(f"{EMBEDDING_URL}/v1/papers", 120, "embedding")
    wait_http(f"{UI_URL}/v1/status", 30, "ui")
    try:
        import requests
        n = len(requests.get(f"{EMBEDDING_URL}/v1/papers", timeout=5).json()["papers"])
    except Exception:
        n = "?"
    say(f"\nReady — {n} papers in the vector store.\n  UI   : {UI_URL}   (first question loads the model: ~10-20s)\n"
        f"  CLI  : cd app/rag_setup && python rag.py\n  stop : anneal stop")
    if a.with_ingest:
        say("  add papers: copy PDFs to app/data/raw_pdfs/ then: python app/scripts/register_pdfs.py")


def cmd_daily_ingest(a):
    """Download one cycle, then process everything pending, then stop what we started."""
    started = []
    if not RegistryClient().health():
        require_ports_free([SERVICE_PORTS["registry"]])
        start("registry")
        started.append("registry")
        if not wait_http(f"{REGISTRY_URL}/v1/health", 30, "registry"):
            sys.exit(1)
    # Topics configured in the UI win over download_manager's built-in DOMAINS.
    # Each is searched in its own run so it can carry its own cap — the
    # downloader's --max applies to every domain in a run, not per domain.
    topics = []
    try:
        from common import settings as settings_store
        sched = settings_store.load().get("ingestion", {}).get("schedule", {})
        topics = [t for t in sched.get("topics", []) if t.get("enabled", True) and t.get("topic")]
    except Exception as e:                                       # noqa: BLE001
        say(f"could not read scheduled topics ({e}); using the built-in domains")

    if topics:
        say(f"== download cycle: {len(topics)} configured topic(s)")
        for t in topics:
            cap = int(t.get("max_papers") or a.max or 10)
            say(f"-- {t['topic']} (up to {cap})")
            subprocess.run([str(VENV_PYTHON), "downloader.py", "--once",
                            "--domain", t["topic"], "--max", str(cap)],
                           cwd=APP_ROOT / "download_manager", check=False)
    else:
        say("== download cycle: built-in domains")
        cmd = [str(VENV_PYTHON), "downloader.py", "--once"] + (["--max", str(a.max)] if a.max is not None else [])
        subprocess.run(cmd, cwd=APP_ROOT / "download_manager", check=False)

    registry = RegistryClient()
    stats = registry.stats()
    pending = stats["total"] - stats["counts"].get("embedded", 0) - stats["counts"].get("error", 0)
    if pending:
        say(f"== processing {pending} pending paper(s)")
        for name, env in (("parse", None), ("embedding", {"EMBED_DEVICE": os.environ.get("EMBED_DEVICE", "cuda")})):
            if not running(name):
                start(name, env)
                started.append(name)
        last_done, last_change = -1, time.time()
        while True:
            time.sleep(30)
            stats = registry.stats()
            done = stats["counts"].get("embedded", 0) + stats["counts"].get("error", 0)
            if done >= stats["total"]:
                break
            if done != last_done:
                last_done, last_change = done, time.time()
            elif time.time() - last_change > a.stall * 60:
                say(f"no progress for {a.stall} min; giving up")
                break
            if not all(running(n) for n in ("parse", "embedding") if n in started):
                say("a service died; see run/logs/")
                break
    say(f"== done: {registry.stats()['counts']}")

    # A parse that dies half way still writes partial output and still reports
    # `embedded`; nothing downstream notices. Check before walking away.
    try:
        from common import coverage
        say("== " + coverage.format_report(coverage.audit()))
    except Exception as e:                                       # noqa: BLE001
        say(f"coverage check failed: {e}")
    if started:
        stop(names=started)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-ingest", action="store_true",
                    help="also run parse/prune, embedder on GPU (with the default command)")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open a browser (with the default command)")
    sub = ap.add_subparsers(dest="cmd")
    u = sub.add_parser("up", help="start everything and open the UI (the default)")
    u.add_argument("--with-ingest", action="store_true")
    u.add_argument("--no-open", action="store_true")
    u.set_defaults(fn=cmd_up)
    sub.add_parser("status", help="what is running and what is indexed").set_defaults(fn=cmd_status)
    f = sub.add_parser("fresh-start"); f.add_argument("--yes", action="store_true"); f.add_argument("--limit", type=int); f.add_argument("--no-ui", action="store_true"); f.set_defaults(fn=cmd_fresh_start)
    q = sub.add_parser("start-query"); q.add_argument("--with-ingest", action="store_true"); q.set_defaults(fn=cmd_start_query)
    s = sub.add_parser("stop"); s.add_argument("--all", action="store_true", help="also sweep orphans by name and port"); s.set_defaults(fn=lambda a: stop(sweep=a.all))
    d = sub.add_parser("daily-ingest"); d.add_argument("--max", type=int, default=None); d.add_argument("--stall", type=int, default=30); d.set_defaults(fn=cmd_daily_ingest)
    st = sub.add_parser("start"); st.add_argument("services", nargs="+", choices=list(SERVICES)); st.set_defaults(fn=lambda a: [start(n) for n in a.services])
    rs = sub.add_parser("restart"); rs.add_argument("services", nargs="+", choices=list(SERVICES)); rs.set_defaults(fn=lambda a: (stop(names=a.services), [start(n) for n in a.services]))
    args = ap.parse_args()
    if args.cmd is None:                     # bare `anneal` == `anneal up`
        args.fn = cmd_up
    if not VENV_PYTHON.exists():
        sys.exit(f"annealenv not found at {VENV_PYTHON}\n"
                 "  python3.12 -m venv virtual_environments/annealenv\n"
                 "  virtual_environments/annealenv/bin/pip install -r app/requirements.txt\n"
                 "  virtual_environments/annealenv/bin/pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2")
    args.fn(args)


if __name__ == "__main__":
    main()
