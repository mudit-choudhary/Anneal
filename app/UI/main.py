"""Web UI service (port 4002).

Serves the React build from UI/static/ (or the legacy single-page
UI/index.html if no build exists) and the JSON API it uses:

    POST /v1/query              {query, filenames?, web?, chat_id?}  -> ndjson stream:
                                {type: chat}, {type: sources}, {type: warning}*, {type: delta}*, {type: done|error}
    GET  /v1/status             ollama / embedding / backend health
    GET  /v1/papers             embedded paper stems (sidebar filter)
    GET  /v1/settings           effective settings (secrets masked)
    PUT  /v1/settings           merge + save
    GET  /v1/ingestion          counts, stage timings, ETA, services, disk
    GET  /v1/logs/stream?service=parse   server-sent events tailing run/logs/<service>.log
    GET  /v1/chats  POST /v1/chats  GET/PATCH/DELETE /v1/chats/{id}  POST /v1/chats/{id}/embed
    DELETE /v1/chats            bulk: {ids:[...]} or {scope:'all'|'unanswered'}
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

UI_DIR = Path(__file__).resolve().parent
APP_ROOT = UI_DIR.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "rag_setup"))


def _service_module(service: str, name: str):
    """Import `name` from a service folder. Each service has its own bare
    `config.py`; drop any cached one so the module binds to its own."""
    import importlib
    folder = str(APP_ROOT / service)
    if folder in sys.path:
        sys.path.remove(folder)
    sys.path.insert(0, folder)
    if name not in sys.modules:
        sys.modules.pop("config", None)
    return importlib.import_module(name)

import requests
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import rag
from common import settings as settings_store
from common.chatstore import ChatStore
from common.logsetup import get_logger
from common.paths import EMBEDDING_URL, LOG_DIR, PARSED_DIR, PDF_DIR, PID_DIR, PROCESSED_DIR, REGISTRY_URL
from common.registry_client import RegistryClient, RegistryUnavailable

log = get_logger("ui")
app = FastAPI(title="ui", version="1")
chats = ChatStore()
STATIC_DIR = UI_DIR / "static"
LEGACY_INDEX = UI_DIR / "index.html"
LOG_SERVICES = ("registry", "parse", "embedding", "prune", "ui", "download", "daily-ingest")


# --------------------------------------------------------------- models
class QueryRequest(BaseModel):
    query: str
    filenames: Optional[List[str]] = None
    web: bool = False
    chat_id: Optional[str] = None


class ChatCreate(BaseModel):
    title: str = "New chat"


class ChatPatch(BaseModel):
    title: str


# --------------------------------------------------------------- status
@app.get("/v1/status")
def status():
    cfg = settings_store.load()
    llm = cfg["llm"]
    ollama_up, model_pulled = False, False
    model = llm["local"].get("model", "")
    try:
        r = requests.get(f"{llm['local'].get('url', 'http://127.0.0.1:11434')}/api/tags", timeout=3)
        ollama_up = r.ok
        models = [m.get("name", "") for m in r.json().get("models", [])]
        model_pulled = model in models or f"{model}:latest" in models
    except requests.RequestException:
        pass
    embeddings_up = False
    try:
        embeddings_up = requests.get(f"{EMBEDDING_URL}/v1/health", timeout=3).ok
    except requests.RequestException:
        pass
    oa = llm.get("openai", {})
    return {
        "backend": llm.get("backend", "local"),
        "ollama": ollama_up, "model": model, "model_pulled": model_pulled,
        "openai_configured": bool(oa.get("base_url") and oa.get("model")),
        "openai_model": oa.get("model", ""),
        "embeddings": embeddings_up,
        "registry": RegistryClient().health(),
    }


@app.get("/v1/papers")
def papers():
    """Embedded papers, each with the date it was embedded so the filter can
    group them by day."""
    try:
        stems = requests.get(f"{EMBEDDING_URL}/v1/papers", timeout=10).json().get("papers", [])
    except requests.RequestException:
        return {"papers": []}
    dates: Dict[str, Any] = {}
    try:
        for p in RegistryClient().list_papers():
            dates[p["filename"]] = p.get("embedded_at") or p.get("downloaded_at")
    except RegistryUnavailable:
        pass
    return {"papers": [{"filename": s, "embedded_at": dates.get(s)} for s in stems]}


# --------------------------------------------------------------- settings
@app.get("/v1/settings")
def get_settings():
    return settings_store.masked(settings_store.load())


@app.put("/v1/settings")
def put_settings(update: Dict[str, Any]):
    current = settings_store.load()
    saved = settings_store.save(settings_store.unmask(update, current))
    log.info("settings updated (backend=%s)", saved["llm"]["backend"])
    return settings_store.masked(saved)


# --------------------------------------------------------------- query
@app.post("/v1/query")
def query(req: QueryRequest):
    question = req.query.strip()
    if not question:
        raise HTTPException(422, "empty query")
    cfg = settings_store.load()

    def event(**kw):
        return json.dumps(kw) + "\n"

    def generate():
        chat_id = req.chat_id
        if chat_id and not chats.get(chat_id):
            chat_id = None
        if not chat_id:
            chat_id = chats.create(question[:80])["id"]
        yield event(type="chat", chat_id=chat_id)
        # Earlier turns of *this* chat, captured before the new question is
        # stored, so a follow-up like "and its limitations?" has a subject.
        existing = chats.get(chat_id) or {"messages": []}
        history = [{"role": m["role"], "content": m["content"]} for m in existing["messages"]]
        chats.add_message(chat_id, "user", question)

        try:
            context, sources, warnings = rag.retrieve(question, req.filenames or None, req.web,
                                                      cfg, history=history)
        except requests.RequestException as e:
            yield event(type="error", message=f"Embedding service unreachable: {e}")
            return
        except Exception as e:
            yield event(type="error", message=f"Retrieval failed: {e}")
            return
        yield event(type="sources", sources=sources, backend=cfg["llm"]["backend"])
        for w in warnings:
            yield event(type="warning", message=w)

        text = ""
        started = time.perf_counter()
        try:
            for delta in rag.answer_stream(context, rag.question_for(question), cfg, history=history):
                text += delta
                yield event(type="delta", text=delta)
        except Exception as e:
            log.exception("generation failed")
            yield event(type="error", message=str(e))
            return
        duration_ms = int((time.perf_counter() - started) * 1000)
        chats.add_message(chat_id, "assistant", text, duration_ms=duration_ms,
                          sources=[{k: v for k, v in s.items() if k != "text"} for s in sources])
        yield event(type="done", duration_ms=duration_ms)

    return StreamingResponse(_with_heartbeat(generate(), event(type="ping")),
                             media_type="application/x-ndjson")


def _with_heartbeat(gen, ping: str, every: float = 2.0):
    """Relay `gen` from a worker thread, sending `ping` whenever it is quiet.

    Retrieval, web fetches and a model still reading its prompt can go silent
    for a minute. Firefox drops a silent connection within seconds of any
    network change (a container starting is enough), so keep bytes flowing.
    If the client leaves, the worker stops at the next item."""
    import queue
    import threading

    items: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    end = object()

    def work():
        try:
            for item in gen:
                if stop.is_set():
                    break
                items.put(item)
        except Exception:                                        # noqa: BLE001
            log.exception("query stream failed")
        finally:
            gen.close()
            items.put(end)

    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            try:
                item = items.get(timeout=every)
            except queue.Empty:
                yield ping
                continue
            if item is end:
                return
            yield item
    finally:
        stop.set()


# --------------------------------------------------------------- chats
@app.get("/v1/chats")
def list_chats():
    return {"chats": chats.list()}


@app.post("/v1/chats")
def create_chat(body: ChatCreate):
    return chats.create(body.title)


@app.get("/v1/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    return chat


@app.patch("/v1/chats/{chat_id}")
def rename_chat(chat_id: str, body: ChatPatch):
    if not chats.rename(chat_id, body.title):
        raise HTTPException(404, "no such chat")
    return {"success": True}


@app.delete("/v1/chats/{chat_id}")
def delete_chat(chat_id: str):
    if not chats.delete(chat_id):
        raise HTTPException(404, "no such chat")
    try:
        requests.delete(f"{EMBEDDING_URL}/v1/chats/{chat_id}", timeout=10)
    except requests.RequestException:
        pass
    return {"success": True}


class ChatBulkDelete(BaseModel):
    ids: Optional[List[str]] = None
    scope: Optional[Literal["all", "unanswered"]] = None


@app.delete("/v1/chats")
def delete_chats(body: ChatBulkDelete):
    """Delete many chats in one call — a selection from the list, or everything.

    Exactly one of `ids` or `scope` is expected; sending both is a caller bug
    worth rejecting rather than guessing at.
    """
    if (body.ids is None) == (body.scope is None):
        raise HTTPException(422, "send either ids or scope, not both")
    targets = body.ids if body.ids is not None else chats.ids(body.scope)
    deleted = chats.delete_many(targets)

    # Only embedded chats have anything in the vector store. Stop after the
    # first connection failure so a stopped embedder cannot turn a 40-chat
    # delete into 40 timeouts.
    embedded = [d["id"] for d in deleted if d["embedded"]]
    orphaned = 0
    for i, chat_id in enumerate(embedded):
        try:
            requests.delete(f"{EMBEDDING_URL}/v1/chats/{chat_id}", timeout=5)
        except requests.RequestException:
            orphaned = len(embedded) - i
            break

    result = {"deleted": len(deleted), "requested": len(targets)}
    if orphaned:
        result["note"] = (f"Deleted, but {orphaned} chat(s) could not be removed from the "
                          f"vector store — the embedding service is not reachable. They stay "
                          f"searchable in memory until it is running and they are deleted again.")
    return result


@app.post("/v1/chats/{chat_id}/embed")
def embed_chat(chat_id: str):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    pairs = chats.qa_pairs(chat_id)
    if not pairs:
        raise HTTPException(422, "chat has no question/answer pairs yet")
    try:
        r = requests.post(f"{EMBEDDING_URL}/v1/chats",
                          json={"chat_id": chat_id, "title": chat["title"],
                                "created_at": chat["created_at"], "pairs": pairs}, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(503, f"embedding service unreachable: {e}")
    chats.mark_embedded(chat_id, True)
    return {"embedded": r.json().get("embedded", 0)}


# --------------------------------------------------------------- fetching papers
class ArxivRequest(BaseModel):
    domain: str
    max_papers: int = 10


class UrlRequest(BaseModel):
    url: str


def _run_downloader(args: List[str]) -> Dict[str, Any]:
    """Launch the downloader detached, logging to run/logs/download.log so the
    Ingestion tab's log view can follow it."""
    import subprocess
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_DIR / "download.log", "a", buffering=1)
    proc = subprocess.Popen(
        [sys.executable, "downloader.py", *args],
        cwd=APP_ROOT / "download_manager", stdout=log_file, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    log.info("downloader started (pid %s): %s", proc.pid, " ".join(args))
    return {"started": True, "pid": proc.pid, "watch": "Ingestion tab -> logs -> download"}


@app.post("/v1/ingest/arxiv")
def ingest_arxiv(req: ArxivRequest):
    """Fetch up to N new papers for a topic. Registered papers flow through
    the normal pipeline, so parse_manager must be running to process them."""
    domain = req.domain.strip()
    if not domain:
        raise HTTPException(422, "domain is required")
    if not RegistryClient().health():
        raise HTTPException(503, "registry is not running — papers could not be registered")
    return _run_downloader(["--once", "--domain", domain, "--max", str(max(1, req.max_papers))])


@app.post("/v1/ingest/url")
def ingest_url(req: UrlRequest):
    """Download one PDF by direct link (arXiv abs/pdf links keep their title)."""
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(422, "url must start with http:// or https://")
    if not RegistryClient().health():
        raise HTTPException(503, "registry is not running — the paper could not be registered")
    return _run_downloader(["--url", url])


# --------------------------------------------------------------- coverage audit
@app.get("/v1/audit")
def audit_coverage():
    """Papers whose parse did not cover the whole document.

    The registry records the stage a paper reached, never whether that stage
    covered all of it, so a parse that died half way still ends up marked
    `embedded`. This is the check for that.
    """
    from common import coverage
    return coverage.audit()


class RepairRequest(BaseModel):
    filenames: List[str]


@app.post("/v1/audit/repair")
def audit_repair(body: RepairRequest):
    """Re-ingest the named papers from their PDFs.

    Exactly the sequence that fixed the two truncated papers by hand: delete
    the partial intermediates so the parser cannot skip them, then send the
    registry row back to `downloaded` so the normal pipeline picks it up.
    Re-embedding drops a filename's old chunks, so the vector store needs no
    separate cleanup.
    """
    import sqlite3

    from common import coverage
    from common.paths import PDF_DIR, REGISTRY_DB

    names = [n.strip() for n in body.filenames if n.strip()]
    if not names:
        raise HTTPException(422, "no filenames given")

    repaired, skipped = [], []
    for stem in names:
        if not (PDF_DIR / f"{stem}.pdf").exists():
            skipped.append({"filename": stem,
                            "why": "the PDF is gone, so it cannot be re-parsed"})
            continue
        coverage.clear_artifacts(stem)
        repaired.append(stem)

    if repaired:
        try:
            con = sqlite3.connect(REGISTRY_DB)
            con.executemany(
                "UPDATE file_status_table SET status='downloaded', parsed_at=NULL, "
                "processed_at=NULL, embedded_at=NULL WHERE filename=?",
                [(s,) for s in repaired])
            con.commit()
            con.close()
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(500, f"could not reset the registry: {e}")

    ops = _ops()
    idle = [n for n in ("parse", "embedding") if not ops.running(n)]
    note = (f"{len(repaired)} paper(s) queued for re-ingest."
            if repaired else "Nothing could be repaired.")
    if repaired and idle:
        note += (f" Start {' and '.join(idle)} from Pipeline control — "
                 "nothing will happen until they are running.")
    if skipped:
        note += f" {len(skipped)} skipped."
    return {"repaired": repaired, "skipped": skipped,
            "services_needed": idle, "note": note}


@app.get("/v1/ingest/arxiv/preview")
def ingest_arxiv_preview(topic: str = Query(...), max_papers: int = Query(10, ge=1, le=50)):
    """What a fetch would actually bring back, without downloading anything.

    Fetching N papers blind is a guess. This runs the same arXiv search the
    downloader runs and returns the candidates so they can be read and chosen
    from; anything already in the registry is marked rather than hidden, so it
    is obvious when a topic is already exhausted.
    """
    import arxiv

    topic = topic.strip()
    if not topic:
        raise HTTPException(422, "topic is required")

    try:
        known = {p["filename"] for p in RegistryClient().list_papers()}
    except Exception:                                            # noqa: BLE001
        known = set()

    sanitize_filename = _service_module("download_manager", "downloader").sanitize_filename

    try:
        client = arxiv.Client(page_size=max_papers, delay_seconds=3.0, num_retries=3)
        search = arxiv.Search(query=f'ti:"{topic}" OR abs:"{topic}"',
                              max_results=max_papers,
                              sort_by=arxiv.SortCriterion.SubmittedDate)
        results = list(client.results(search))
    except Exception as e:                                       # noqa: BLE001
        raise HTTPException(502, f"arXiv search failed: {e}")

    papers = []
    for r in results:
        stem = sanitize_filename(r.title)
        papers.append({
            "title": r.title.strip(),
            "authors": [a.name for a in r.authors][:6],
            "published": r.published.isoformat() if r.published else None,
            "summary": (r.summary or "").strip().replace("\n", " ")[:400],
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
            "categories": list(r.categories)[:4],
            "already_have": stem in known,
        })
    return {"topic": topic, "papers": papers}


class PickRequest(BaseModel):
    urls: List[str]


@app.post("/v1/ingest/pick")
def ingest_pick(req: PickRequest):
    """Download a chosen set of papers rather than the first N of a search."""
    urls = [u.strip() for u in req.urls if u.strip().startswith(("http://", "https://"))]
    if not urls:
        raise HTTPException(422, "no valid urls")
    if len(urls) > 50:
        raise HTTPException(422, "at most 50 papers at a time")
    if not RegistryClient().health():
        raise HTTPException(503, "registry is not running — papers could not be registered")
    args = []
    for u in urls:
        args += ["--url", u]
    result = _run_downloader(args)
    result["count"] = len(urls)
    return result


# --------------------------------------------------------------- schedule
SCHED_UNIT = "rag-daily-ingest.timer"
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"


def _systemctl(*args, timeout=10):
    """Best-effort systemctl --user. Returns (ok, output)."""
    import subprocess
    try:
        p = subprocess.run(["systemctl", "--user", *args], capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout or p.stderr).strip()
    except Exception as e:                                       # noqa: BLE001
        return False, str(e)


@app.get("/v1/schedule")
def get_schedule():
    """The configured topics plus the real state of the systemd timer."""
    cfg = settings_store.load().get("ingestion", {}).get("schedule", {})
    installed = (UNIT_DIR / SCHED_UNIT).exists()
    enabled, next_run, last_run = False, None, None
    if installed:
        ok, out = _systemctl("is-enabled", SCHED_UNIT)
        enabled = ok and out.strip() == "enabled"
        ok, out = _systemctl("show", SCHED_UNIT, "-p", "NextElapseUSecRealtime",
                             "-p", "LastTriggerUSec")
        if ok:
            for line in out.splitlines():
                key, _, value = line.partition("=")
                value = value.strip()
                if not value or value in ("0", "n/a"):
                    continue
                if key == "NextElapseUSecRealtime":
                    next_run = value
                elif key == "LastTriggerUSec":
                    last_run = value
    # lingering decides whether the timer survives logout; it needs sudo to set
    linger = False
    try:
        import subprocess
        p = subprocess.run(["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
                           capture_output=True, text=True, timeout=5)
        linger = "Linger=yes" in (p.stdout or "")
    except Exception:                                            # noqa: BLE001
        pass
    return {"time": cfg.get("time", "03:00"), "topics": cfg.get("topics", []),
            "installed": installed, "enabled": enabled,
            "next_run": next_run, "last_run": last_run, "linger": linger}


class ScheduleTopic(BaseModel):
    topic: str
    max_papers: int = 10
    enabled: bool = True


class SchedulePut(BaseModel):
    time: Optional[str] = None
    topics: Optional[List[ScheduleTopic]] = None
    enabled: Optional[bool] = None


@app.put("/v1/schedule")
def put_schedule(body: SchedulePut):
    """Save the topic list, and install/enable the timer to match.

    The topics live in settings so `ops.py daily-ingest` picks them up with no
    unit change; only the time of day is baked into the timer file, so the
    units are only rewritten when that changes or they are missing.
    """
    import re as _re
    import subprocess

    # Merge only the schedule subtree: save() deep-merges, so writing the whole
    # config back would bake every default into the file for no reason.
    sched = dict(settings_store.load().get("ingestion", {}).get("schedule", {}))

    if body.time is not None:
        if not _re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", body.time):
            raise HTTPException(422, "time must be HH:MM in 24-hour form")
        sched["time"] = body.time
    if body.topics is not None:
        cleaned = []
        for t in body.topics:
            name = t.topic.strip()
            if name:
                cleaned.append({"topic": name,
                                "max_papers": max(1, min(200, t.max_papers)),
                                "enabled": t.enabled})
        sched["topics"] = cleaned
    settings_store.save({"ingestion": {"schedule": sched}})

    notes = []
    want_enabled = body.enabled
    need_install = body.time is not None or not (UNIT_DIR / SCHED_UNIT).exists()
    if need_install or want_enabled:
        installer = APP_ROOT / "ops" / "systemd" / "install.sh"
        if not installer.exists():
            notes.append("the unit installer is missing; the timer was not changed")
        else:
            try:
                p = subprocess.run(["bash", str(installer), "--time", sched.get("time", "03:00")],
                                   capture_output=True, text=True, timeout=30)
                if p.returncode != 0:
                    notes.append(f"installing the timer failed: {(p.stderr or p.stdout).strip()[:200]}")
            except Exception as e:                               # noqa: BLE001
                notes.append(f"installing the timer failed: {e}")
            _systemctl("daemon-reload")

    if want_enabled is True:
        ok, out = _systemctl("enable", "--now", SCHED_UNIT)
        # Persistent=true treats "never run" as a missed run, so switching the
        # timer on starts a catch-up cycle straight away. That is the behaviour
        # that makes a missed overnight run fire after boot, but it surprises
        # people the first time, so say it rather than let downloads appear.
        notes.append("Daily downloads are on — a first catch-up run starts now."
                     if ok else f"could not enable the timer: {out[:200]}")
    elif want_enabled is False:
        ok, out = _systemctl("disable", "--now", SCHED_UNIT)
        notes.append("Daily downloads are off." if ok else f"could not disable the timer: {out[:200]}")

    state = get_schedule()
    if want_enabled is not False and state["enabled"] and not state["linger"]:
        notes.append("The timer only runs while you are logged in. To keep it running "
                     "after logout: loginctl enable-linger $USER")
    state["note"] = " ".join(notes) if notes else None
    return state


@app.get("/v1/prune/candidates")
def prune_candidates():
    """Embedded papers whose PDF is still on disk, oldest first.

    This is exactly the list the keep-strategy slices. The UI gets it once and
    works out the boundary itself, so dragging the "spare" control redraws
    instantly instead of asking the server on every step.
    """
    pdf_dir = APP_ROOT / "data" / "raw_pdfs"
    try:
        papers = RegistryClient().list_papers(status="embedded")
    except Exception:                                        # noqa: BLE001
        return {"papers": [], "registry": False}

    out = []
    for p in papers:
        pdf = pdf_dir / f"{p['filename']}.pdf"
        if not pdf.exists():
            continue
        out.append({"filename": p["filename"],
                    "downloaded_at": p.get("downloaded_at"),
                    "bytes": pdf.stat().st_size})
    out.sort(key=lambda x: x["downloaded_at"] or "")
    return {"papers": out, "registry": True}


@app.post("/v1/prune/run")
def prune_run(dry_run: bool = Query(True), prune: Optional[Dict[str, Any]] = Body(None)):
    """Sweep now. Defaults to a dry run so the UI can show what *would* be
    removed before anything is deleted. A preview may pass the unsaved
    settings on screen; they are merged over the saved ones, never stored."""
    pruning = _service_module("prune_manager", "pruning")
    cfg = {**settings_store.load()["prune"], **(prune or {})}
    try:
        counts = pruning.sweep(RegistryClient(), cfg, dry_run=dry_run)
    except RegistryUnavailable as e:
        raise HTTPException(503, str(e))
    return {"dry_run": dry_run, "counts": counts}


# --------------------------------------------------------------- GPU
def _nvidia_smi(query: str, mode: str = "gpu") -> List[List[str]]:
    import subprocess
    flag = "--query-gpu" if mode == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return []
    if out.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


def _ollama_placement(url: str) -> Optional[Dict[str, Any]]:
    """`ollama ps` is the only place the CPU/GPU layer split is reported."""
    import re
    import subprocess
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None
    rows = [l for l in out.splitlines()[1:] if l.strip()]
    if not rows:
        return None
    parts = rows[0].split()
    split = next((p for p in parts if re.fullmatch(r"\d+%/\d+%", p)), None)
    gpu_pct = int(split.split("/")[1].rstrip("%")) if split else None
    size = next((f"{parts[i]} {parts[i + 1]}" for i in range(len(parts) - 1)
                 if re.fullmatch(r"[\d.]+", parts[i]) and parts[i + 1] in ("GB", "MB")), None)
    return {"model": parts[0], "size": size, "split": split, "gpu_percent": gpu_pct,
            "loaded": True}


@app.get("/v1/gpu")
def gpu():
    """VRAM usage and where each of our three models is actually running."""
    ops = _ops()
    cfg = settings_store.load()

    info = _nvidia_smi("name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu")
    device = None
    if info:
        n, total, used, free, util, temp = info[0][:6]
        device = {"name": n, "total_mb": int(total), "used_mb": int(used), "free_mb": int(free),
                  "utilisation_percent": int(util), "temperature_c": int(temp)}

    # map GPU processes back to our services where we can
    owners = {}
    for name in ops.SERVICES:
        pid = ops.pid_of(name)
        if pid:
            owners[pid] = name
    processes = []
    for row in _nvidia_smi("pid,used_memory", mode="apps"):
        try:
            pid, mem = int(row[0]), int(row[1])
        except (ValueError, IndexError):
            continue
        label = owners.get(pid)
        if not label:
            try:
                import psutil
                label = psutil.Process(pid).name()
            except Exception:
                label = "unknown"
        processes.append({"pid": pid, "used_mb": mem, "owner": label})
    processes.sort(key=lambda p: -p["used_mb"])

    # --- our three models ---
    models = []
    placement = _ollama_placement(cfg["llm"]["local"].get("url", ""))
    models.append({
        "name": "Answering model (Ollama)",
        "detail": cfg["llm"]["local"].get("model", ""),
        "placement": placement["split"] if placement else "not loaded",
        "gpu_percent": placement["gpu_percent"] if placement else 0,
        "note": ("Ollama offloads whole transformer layers; the split is chosen when the model "
                 "loads, from whatever VRAM is free at that moment."),
        "splittable": True,
    })

    embed_device, embed_model = None, None
    try:
        h = requests.get(f"{EMBEDDING_URL}/v1/health", timeout=3).json()
        embed_device, embed_model = h.get("device"), h.get("model")
    except requests.RequestException:
        pass
    models.append({
        "name": "Embedder (bge-base)",
        "detail": embed_model or "",
        "placement": embed_device or "not running",
        "gpu_percent": 100 if embed_device == "cuda" else 0,
        "note": ("All-or-nothing: PyTorch loads it on one device. It is only ~440MB, so a split "
                 "would cost more in transfers than it saves."),
        "splittable": False,
    })

    provider = None
    parse_log = LOG_DIR / "parse.log"
    if parse_log.exists():
        try:
            for line in reversed(parse_log.read_text(errors="replace").splitlines()[-400:]):
                if "layout model on" in line:
                    provider = line.rsplit("layout model on", 1)[1].strip()
                    break
        except OSError:
            pass
    models.append({
        "name": "Layout model (YOLOv11)",
        "detail": "onnxruntime",
        "placement": (provider or "not started").replace("ExecutionProvider", ""),
        "gpu_percent": 100 if provider and "CUDA" in provider else 0,
        "note": ("onnxruntime places each operator, falling back to CPU for anything the GPU "
                 "provider cannot run — automatic, and not something you can dial."),
        "splittable": "partial",
    })

    return {"available": device is not None, "device": device,
            "processes": processes, "models": models}


# --------------------------------------------------------------- service control
def _ops():
    """scripts/ops.py holds all the process management; reuse it rather than
    shelling out to the wrapper scripts."""
    sys.path.insert(0, str(APP_ROOT / "scripts"))
    import importlib
    return importlib.import_module("ops")


class ServiceAction(BaseModel):
    action: str                      # start | stop | restart
    embed_device: Optional[str] = None   # "cuda" for ingestion, "cpu" for querying


@app.get("/v1/services")
def list_services():
    """Every manageable service, its state, and what it is for."""
    import psutil
    ops = _ops()
    out = []
    for name, (title, purpose, port) in ops.SERVICE_INFO.items():
        pid = ops.pid_of(name)
        alive = bool(pid and psutil.pid_exists(pid))
        uptime = None
        if alive:
            try:
                uptime = int(time.time() - psutil.Process(pid).create_time())
            except psutil.Error:
                alive = False
        log = LOG_DIR / f"{name}.log"
        out.append({
            "name": name, "title": title, "purpose": purpose, "port": port,
            "running": alive, "pid": pid if alive else None, "uptime_seconds": uptime,
            # Stopping the UI from the UI would kill the request mid-flight and
            # leave no way back in except the terminal.
            "self": name == "ui",
            "log_bytes": log.stat().st_size if log.exists() else 0,
        })
    return {"services": out}


@app.post("/v1/services/{name}")
def control_service(name: str, body: ServiceAction):
    ops = _ops()
    if name not in ops.SERVICES:
        raise HTTPException(404, f"unknown service; one of {sorted(ops.SERVICES)}")
    if body.action not in ("start", "stop", "restart"):
        raise HTTPException(422, "action must be start, stop or restart")
    if name == "ui" and body.action in ("stop", "restart"):
        raise HTTPException(
            409, "The web UI cannot stop itself — this page would go with it. "
                 "Use `scripts/stop_services.sh` or `python scripts/ops.py restart ui`.")

    env = {}
    if name == "embedding":
        device = body.embed_device or ("cuda" if ops.running("parse") else "cpu")
        env["EMBED_DEVICE"] = device

    try:
        if body.action in ("stop", "restart"):
            ops.stop(names=[name])
        if body.action in ("start", "restart"):
            if ops.running(name):
                return {"name": name, "running": True, "note": "already running"}
            port = ops.SERVICE_INFO[name][2]
            if port and ops.port_owner(port):
                raise HTTPException(409, f"port {port} is already in use by another process")
            ops.start(name, env or None)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("service control failed: %s %s", body.action, name)
        raise HTTPException(500, f"{body.action} failed: {e}")

    log.info("%s %s via UI%s", body.action, name, f" (EMBED_DEVICE={env['EMBED_DEVICE']})" if env else "")
    result = {"name": name, "action": body.action, "running": ops.running(name), "env": env or None}

    # The embedder silently starts on the CPU when the GPU is full, so asking
    # for 'cuda' and getting CPU must be *reported*, not left for the user to
    # discover on the GPU tab. Wait for it to finish loading and check.
    if name == "embedding" and body.action in ("start", "restart"):
        wanted = env.get("EMBED_DEVICE")
        actual = None
        for _ in range(60):
            time.sleep(1)
            try:
                actual = requests.get(f"{EMBEDDING_URL}/v1/health", timeout=2).json().get("device")
                break
            except requests.RequestException:
                if not ops.running(name):
                    break
        result["device"] = actual
        if actual and wanted and actual != wanted:
            free = _free_vram_mb()
            result["note"] = (
                f"Started on {actual.upper()}, not {wanted.upper()} — there was not enough free "
                f"VRAM when the model loaded"
                + (f" ({free} MB free)." if free is not None else ".")
                + " Free some (stop the answering model with `ollama stop`, or the parser) and "
                  "restart the embedder.")
        elif actual:
            result["note"] = f"Embedder is running on {actual.upper()}."
    return result


def _free_vram_mb() -> Optional[int]:
    rows = _nvidia_smi("memory.free")
    try:
        return int(rows[0][0])
    except (IndexError, ValueError):
        return None


@app.post("/v1/services/preset/{preset}")
def apply_preset(preset: str):
    """Start the usual combinations without hunting for the right shell script.

    query   — answer questions: registry + embedder on CPU (the GPU stays
              free for the LLM)
    ingest  — also process new papers: + parser + pruner, embedder on GPU
    """
    ops = _ops()
    if preset not in ("query", "ingest"):
        raise HTTPException(404, "preset must be 'query' or 'ingest'")

    wanted = ["registry", "embedding"] + (["parse", "prune"] if preset == "ingest" else [])
    device = "cuda" if preset == "ingest" else "cpu"
    started, already = [], []
    for name in wanted:
        if ops.running(name):
            already.append(name)
            continue
        port = ops.SERVICE_INFO[name][2]
        if port and ops.port_owner(port):
            already.append(name)          # something is already serving it
            continue
        ops.start(name, {"EMBED_DEVICE": device} if name == "embedding" else None)
        started.append(name)

    # The embedder's device is fixed at startup, so a running one on the wrong
    # device has to be restarted for the preset to mean anything.
    note = None
    if "embedding" in already and preset == "ingest":
        note = ("The embedder is already running — restart it with device 'cuda' "
                "if you want it on the GPU for ingestion.")
    log.info("preset %s: started %s, already up %s", preset, started, already)
    return {"preset": preset, "started": started, "already_running": already,
            "embed_device": device, "note": note}


# --------------------------------------------------------------- ingestion
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@app.get("/v1/ingestion")
def ingestion():
    services = {}
    if PID_DIR.exists():
        for pidfile in PID_DIR.glob("*.pid"):
            try:
                pid = int(pidfile.read_text().strip())
            except ValueError:
                continue
            services[pidfile.stem] = {"pid": pid, "running": _alive(pid)}
    # Files on disk are *not* the same thing as pipeline status: a paper can be
    # fully embedded while its intermediates are still on disk waiting for the
    # pruner. Report papers as well as files, because data/processed holds two
    # files per paper (.txt and .json) and "96" is otherwise baffling.
    def disk(directory):
        if not directory.exists():
            return {"files": 0, "papers": 0, "bytes": 0}
        entries = [p for p in directory.glob("*") if p.is_file()]
        return {"files": len(entries),
                "papers": len({p.stem.replace(".blocks", "") for p in entries}),
                "bytes": sum(p.stat().st_size for p in entries)}

    out: Dict[str, Any] = {
        "services": services, "registry": None, "eta_seconds": None, "eta_note": None,
        "throughput_per_hour": None, "pending_bytes": None,
        "files": {"raw_pdfs": disk(PDF_DIR), "parsed": disk(PARSED_DIR),
                  "processed": disk(PROCESSED_DIR)},
    }
    try:
        stats = RegistryClient().stats()
        papers = RegistryClient().list_papers()
    except RegistryUnavailable:
        out["eta_note"] = "registry is not running"
        return out

    out["registry"] = stats
    counts = stats["counts"]
    pending = [p for p in papers if p["status"] in ("downloaded", "parsed", "processed")]
    out["remaining"] = len(pending)

    # How much data is actually left to chew through.
    total_bytes = 0
    for p in pending:
        pdf = PDF_DIR / f"{p['filename']}.pdf"
        if pdf.exists():
            total_bytes += pdf.stat().st_size
    out["pending_bytes"] = total_bytes

    if not pending:
        return out

    # Which worker each pending paper is waiting on — an ETA is meaningless
    # if nothing is running to do the work.
    needs_parse = any(p["status"] in ("downloaded", "parsed") for p in pending)
    needs_embed = any(p["status"] == "processed" for p in pending)
    idle = [name for name, needed in (("parse", needs_parse), ("embedding", needs_embed))
            if needed and not services.get(name, {}).get("running")]
    if idle:
        out["eta_note"] = (f"not progressing — {' and '.join(idle)} "
                           f"{'is' if len(idle) == 1 else 'are'} not running "
                           f"(start with anneal --with-ingest)")
        return out

    # Measured throughput beats modelled per-stage times, which include queue
    # wait. Widen the window until there are enough completions to divide by.
    now = datetime.now(timezone.utc)
    embedded_times = sorted(datetime.fromisoformat(t) for t in stats.get("recent_embedded_at", []))
    for window in (900, 3600, 6 * 3600):
        recent = [t for t in embedded_times if (now - t).total_seconds() < window]
        if len(recent) >= 2:
            span = (max(recent) - min(recent)).total_seconds() or 1
            per_second = len(recent) / span
            out["throughput_per_hour"] = round(per_second * 3600, 1)
            out["eta_seconds"] = int(len(pending) / per_second)
            return out

    out["eta_note"] = "estimating — waiting for the first papers to finish"
    return out


@app.get("/v1/logs/stream")
async def logs_stream(service: str = Query("parse"), lines: int = Query(100, ge=0, le=2000)):
    if service not in LOG_SERVICES:
        raise HTTPException(404, f"unknown service; one of {LOG_SERVICES}")
    path = LOG_DIR / f"{service}.log"

    async def events():
        pos = 0
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-lines:] if lines else []
                for line in tail:
                    yield {"data": line.rstrip("\n")}
                f.seek(0, os.SEEK_END)
                pos = f.tell()
        while True:
            await asyncio.sleep(1)
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                for line in f:
                    yield {"data": line.rstrip("\n")}
                pos = f.tell()

    return EventSourceResponse(events())


# --------------------------------------------------------------- static
if (STATIC_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = STATIC_DIR / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(LEGACY_INDEX)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4002, reload=False, log_level="warning")
