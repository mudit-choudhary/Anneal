"""User-editable runtime settings (LLM backend, retrieval knobs), stored as
JSON in data/settings.json and editable from the UI.

Resolution order: DEFAULTS ← environment variables ← settings file. So a
fresh checkout works with no file, env vars still work for scripting, and
whatever the user saves in the UI wins.
"""

import copy
import json
import os
from typing import Any, Dict

from common.paths import SETTINGS_FILE

DEFAULTS: Dict[str, Any] = {
    "llm": {
        "backend": "local",                      # "local" (Ollama) | "openai" (any OpenAI-compatible API, e.g. OGWY)
        "local": {
            "url": "http://127.0.0.1:11434",
            "model": "qwen3:4b-instruct",
            "num_ctx": 6144,
            "keep_alive": "30m",
            "temperature": 0.2,
        },
        "openai": {
            "base_url": "",                      # e.g. http://127.0.0.1:8080/v1
            "api_key": "",
            "model": "",
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": True,                      # if the API rejects streaming, the answer is sent whole
        },
    },
    "ingestion": {
        # arXiv crawl defaults, used by the UI's "fetch papers" action
        "domain": "Retrieval Augmented Generation",
        "max_papers": 10,
        # Unattended daily crawl, run by the systemd timer through
        # `ops.py daily-ingest`. Each topic is searched separately so it can
        # carry its own cap; an empty list falls back to download_manager's
        # built-in DOMAINS.
        "schedule": {
            "time": "03:00",                 # local time the timer fires
            "topics": [],                    # [{"topic": str, "max_papers": int, "enabled": bool}]
        },
    },
    "prune": {
        # What happens to a raw PDF once its paper is embedded.
        #   keep    — leave it in data/raw_pdfs (default; the only input a
        #             re-ingest can be rebuilt from)
        #   archive — move it to archive_dir
        #   delete  — remove it (irreversible)
        "raw_pdf_policy": "keep",
        "archive_dir": "",
        # Which embedded PDFs stay behind when archiving/deleting:
        #   all — none stay; newest/oldest — keep that many by download time
        "keep_strategy": "all",
        "keep_count": 50,
        "interval_seconds": 1800,
    },
    "retrieval": {
        "n_results": 6,                          # paper chunks handed to the LLM
        "n_results_with_web": 4,                 # fewer when web pages share the context window
        "use_chats": True,                       # also search saved conversations
        "n_chat_results": 2,
        "web_results": 3,                        # pages fetched when web search is on
        "web_chars_per_page": 2000,
    },
}

_ENV_OVERRIDES = {
    ("llm", "backend"): "LLM_BACKEND",
    ("llm", "local", "url"): "OLLAMA_URL",
    ("llm", "local", "model"): "OLLAMA_MODEL",
    ("llm", "openai", "base_url"): "OPENAI_BASE_URL",
    ("llm", "openai", "api_key"): "OPENAI_API_KEY",
    ("llm", "openai", "model"): "OPENAI_MODEL",
}

SECRET_KEYS = {"api_key"}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_path(d: Dict[str, Any], path, value):
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def load() -> Dict[str, Any]:
    settings = copy.deepcopy(DEFAULTS)
    for path, env in _ENV_OVERRIDES.items():
        if os.environ.get(env):
            _set_path(settings, path, os.environ[env])
    if SETTINGS_FILE.exists():
        try:
            settings = _deep_merge(settings, json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save(update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `update` into the stored settings and return the new effective settings."""
    stored = {}
    if SETTINGS_FILE.exists():
        try:
            stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
    merged = _deep_merge(stored, update)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return load()


def masked(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Copy safe to send to a browser: secrets replaced by a marker."""
    out = copy.deepcopy(settings)

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif k in SECRET_KEYS and v:
                d[k] = "••••" + str(v)[-4:]
    walk(out)
    return out


def unmask(update: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Undo `masked` on an update coming back from the browser: a masked
    secret means 'unchanged'."""
    out = copy.deepcopy(update)

    def walk(d, cur):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                walk(v, (cur or {}).get(k, {}))
            elif k in SECRET_KEYS and isinstance(v, str) and v.startswith("••••"):
                d[k] = (cur or {}).get(k, "")
    walk(out, current)
    return out
