"""Paths and service URLs.

Code paths come from this file's location, so every service and script agrees
regardless of its working directory. Everything the app *writes* lives under a
separate data home instead, so the code folder can be read-only (and, one day,
installed): papers, models, the vector store, databases, logs and pids.

    ANNEAL_HOME=/some/where     overrides it (tests, a second corpus, a drive)
    ~/.local/share/anneal       otherwise

Ports come from the environment the same way, so a second instance, or a
machine where 4000-4002 are taken, needs no code change:

    ANNEAL_UI_PORT, ANNEAL_REGISTRY_PORT, ANNEAL_EMBEDDING_PORT

`anneal --port N` (and --registry-port / --embedding-port) set these for every
service it starts; children inherit them, so all three agree.

XDG_DATA_HOME is deliberately ignored: snap-confined terminals (VS Code, the
snap Firefox) set it to a private per-snap directory, so honouring it would
give the app a different corpus depending on which terminal started it.
"""

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent      # app/: the service code
PROJECT_ROOT = APP_ROOT.parent                         # the repository: app/, evals/, tests/


def _data_home() -> Path:
    override = os.environ.get("ANNEAL_HOME")
    return Path(override).expanduser() if override else Path.home() / ".local" / "share" / "anneal"


DATA_HOME = _data_home()                               # everything the app writes

DATA_DIR = DATA_HOME / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
PARSED_DIR = DATA_DIR / "parsed"
PROCESSED_DIR = DATA_DIR / "processed"
DEBUG_DIR = DATA_DIR / "debug"
SETTINGS_FILE = DATA_DIR / "settings.json"
APP_DB = DATA_DIR / "app.db"                 # chats (UI service)
REGISTRY_DB = DATA_HOME / "registry" / "rag_registry.db"
VECTOR_DB = DATA_HOME / "vector_db"
MODELS_DIR = DATA_HOME / "models"
TRAINING_RUNS = DATA_HOME / "runs"           # YOLO fine-tuning output

RUN_DIR = DATA_HOME / "run"
LOG_DIR = RUN_DIR / "logs"
PID_DIR = RUN_DIR / "pids"

VENV_PYTHON = PROJECT_ROOT / "virtual_environments" / "annealenv" / "bin" / "python"

def _port(var: str, default: int) -> int:
    """A port from the environment. A bad value is worth a loud failure: the
    alternative is a service quietly listening somewhere nobody is calling."""
    raw = os.environ.get(var)
    if raw is None or not raw.strip():
        return default
    try:
        port = int(raw)
    except ValueError:
        raise ValueError(f"{var}={raw!r} is not a port number") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"{var}={port} is outside 1-65535")
    return port


REGISTRY_PORT = _port("ANNEAL_REGISTRY_PORT", 4000)
EMBEDDING_PORT = _port("ANNEAL_EMBEDDING_PORT", 4001)
UI_PORT = _port("ANNEAL_UI_PORT", 4002)

REGISTRY_URL = f"http://127.0.0.1:{REGISTRY_PORT}"
EMBEDDING_URL = f"http://127.0.0.1:{EMBEDDING_PORT}"
UI_URL = f"http://127.0.0.1:{UI_PORT}"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

SERVICE_PORTS = {"registry": REGISTRY_PORT, "embedding": EMBEDDING_PORT, "ui": UI_PORT}
