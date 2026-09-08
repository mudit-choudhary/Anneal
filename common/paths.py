"""Repo-relative paths and service URLs, derived from this file's location so
every service and script agrees regardless of its working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
PARSED_DIR = DATA_DIR / "parsed"
PROCESSED_DIR = DATA_DIR / "processed"
DEBUG_DIR = DATA_DIR / "debug"
SETTINGS_FILE = DATA_DIR / "settings.json"
APP_DB = DATA_DIR / "app.db"                 # chats (UI service)
REGISTRY_DB = REPO_ROOT / "registry_manager" / "rag_registry.db"
VECTOR_DB = REPO_ROOT / "vector_db"
MODELS_DIR = REPO_ROOT / "models"

RUN_DIR = REPO_ROOT / "run"
LOG_DIR = RUN_DIR / "logs"
PID_DIR = RUN_DIR / "pids"

VENV_PYTHON = REPO_ROOT / "virtual_environments" / "globalragsetup_env" / "bin" / "python"

REGISTRY_URL = "http://127.0.0.1:4000"
EMBEDDING_URL = "http://127.0.0.1:4001"
UI_URL = "http://127.0.0.1:4002"
OLLAMA_URL = "http://127.0.0.1:11434"

SERVICE_PORTS = {"registry": 4000, "embedding": 4001, "ui": 4002}
