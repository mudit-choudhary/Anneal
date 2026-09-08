"""Report this machine's GPU/RAM and which model choices fit it.

    python scripts/hardware_check.py

The pipeline time-shares one GPU: YOLO + the embedder during ingestion,
the LLM during querying. Recommendations below assume that split.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil

TIERS = [
    # (min VRAM GB, LLM, embedder placement during queries, notes)
    (0,  "qwen3:1.7b or CPU-only qwen3:4b (slow)", "cpu", "no usable GPU: everything on CPU; ingestion is slow but works"),
    (3.5, "qwen3:4b-instruct (Q4, ~3.5 GB)", "cpu", "current setup: LLM alone on the GPU by day; YOLO + bge-base by night"),
    (7.5, "qwen3:8b (Q4, ~5.5 GB)", "cuda", "embedder can stay on the GPU next to the LLM; YOLO batch can double"),
    (11.5, "qwen3:14b (Q4, ~9.5 GB)", "cuda", "all three models resident at once; no day/night split needed"),
    (23, "qwen3:32b (Q4, ~20 GB) or qwen3:14b at Q8", "cuda", "room for a VLM pass over figures/tables as well"),
]


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"name": torch.cuda.get_device_name(0), "total_gb": total / 1e9, "free_gb": free / 1e9,
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))}
    except Exception:
        return None


def main():
    ram = psutil.virtual_memory()
    print(f"RAM      {ram.total / 1e9:.1f} GB total, {ram.available / 1e9:.1f} GB available")
    gpu = gpu_info()
    if gpu:
        print(f"GPU      {gpu['name']} — {gpu['total_gb']:.1f} GB total, {gpu['free_gb']:.1f} GB free now "
              f"(compute {gpu['capability']})")
        vram = gpu["total_gb"]
    else:
        print("GPU      none usable by PyTorch (CUDA unavailable)")
        vram = 0
    tier = max((t for t in TIERS if vram >= t[0]), key=lambda t: t[0])
    print()
    print(f"Fits this machine ({vram:.1f} GB VRAM):")
    print(f"  answering LLM      {tier[1]}")
    print(f"  embedder (queries) BAAI/bge-base-en-v1.5 on {tier[2]}   (EMBED_DEVICE={tier[2]})")
    print(f"  layout model       YOLOv11-small at imgsz 1024, batch {4 if vram < 7.5 else 8}")
    print(f"  note               {tier[3]}")
    print()
    print("Apply: set OLLAMA_MODEL / EMBED_DEVICE in the environment or in the UI settings;")
    print("       `ollama pull <model>` for a different LLM.")


if __name__ == "__main__":
    main()
