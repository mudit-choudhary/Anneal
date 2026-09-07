import os

# --- Backend selection: "local" (Ollama) or "gemini" ---
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")

# --- Local backend (Ollama) ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# Use a non-thinking *instruct* variant: the hybrid qwen3:4b ignores
# think=False and leaks chain-of-thought into streamed answers.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct")
# Context window: ~3-4K tokens of chunks + question fits comfortably; 6144
# (rather than 8192) lets the model + KV cache stay fully on the 4GB GPU.
OLLAMA_NUM_CTX = 6144
# How long Ollama keeps the model loaded after the last query. The daemon
# frees VRAM after this, which is what lets YOLO/embedding jobs use the GPU
# overnight without a manual stop.
OLLAMA_KEEP_ALIVE = "30m"

# --- Gemini backend (higher-quality fallback for multi-paper synthesis) ---
# Set via environment: export GEMINI_API_KEY=...
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# --- Retrieval ---
EMBEDDING_URL = "http://127.0.0.1:4001"
N_RESULTS = 6  # keep retrieval tight: a 4B model degrades with too many chunks
