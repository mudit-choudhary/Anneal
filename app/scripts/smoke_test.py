"""End-to-end smoke test: one command that exercises every stage and says
what works.

    python scripts/smoke_test.py                  # everything it can reach
    python scripts/smoke_test.py --pdf Some.pdf   # use a specific paper
    python scripts/smoke_test.py --offline        # stages 1-3 only, no services
    python scripts/smoke_test.py -v               # show sample output per stage

Stages 1-3 (parse, chunk, embed) need no services and touch **no** project
state: parsing writes to data/debug/smoke/, and embedding goes to a
throwaway vector store in a temp directory. Stages 4-7 (retrieval, answer,
UI, chat memory) use the running services and are skipped with a reason if
they are not up.

Exit code 0 if nothing failed (skips are not failures), 1 otherwise.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from common.paths import DEBUG_DIR, EMBEDDING_URL, PDF_DIR, REGISTRY_URL, UI_URL  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, SKIP = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}SKIP{RESET}"


class Runner:
    def __init__(self, verbose=False):
        self.results = []
        self.verbose = verbose

    def stage(self, name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            status, note = OK, detail or ""
        except SkipStage as e:
            status, note = SKIP, str(e)
        except Exception as e:
            status, note = FAIL, f"{type(e).__name__}: {e}"
            if self.verbose:
                traceback.print_exc()
        elapsed = time.perf_counter() - start
        self.results.append((name, status, note))
        print(f"  {status}  {name:<34} {DIM}{elapsed:5.1f}s{RESET}  {note}")
        return status is OK

    def summary(self):
        passed = sum(1 for _, s, _ in self.results if s is OK)
        failed = [n for n, s, _ in self.results if s is FAIL]
        skipped = [n for n, s, _ in self.results if s is SKIP]
        print(f"\n{'=' * 72}")
        print(f"{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
        if skipped:
            print(f"{YELLOW}skipped{RESET}: {', '.join(skipped)}")
        if failed:
            print(f"{RED}failed{RESET}: {', '.join(failed)}")
        return 1 if failed else 0


class SkipStage(Exception):
    """Raised when a stage's prerequisite is absent — not a failure."""


# Each manager has its own flat `config` module, so importing from a second
# one in the same process would pick up the first's. Clear the shared names
# and put the wanted package first on the path before each stage imports.
FLAT_MODULES = ("config", "chunking", "embeddings", "pdf_parser", "txt_processor",
                "layout_detector", "rag", "websearch")


def use_package(name):
    for mod in FLAT_MODULES:
        sys.modules.pop(mod, None)
    path = str(APP_ROOT / name)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def service_up(url, timeout=2):
    import requests
    try:
        return requests.get(url, timeout=timeout).ok
    except requests.RequestException:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", help="paper to use (default: first in data/raw_pdfs)")
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--offline", action="store_true", help="stages 1-3 only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.pdf:
        pdf = Path(args.pdf) if Path(args.pdf).exists() else PDF_DIR / Path(args.pdf).name
    elif pdfs:
        pdf = pdfs[0]
    else:
        sys.exit(f"No PDFs in {PDF_DIR} — add one or pass --pdf")
    if not pdf.exists():
        sys.exit(f"{pdf} not found")

    work = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
    debug_dir = DEBUG_DIR / "smoke"
    debug_dir.mkdir(parents=True, exist_ok=True)
    # Point the embedder at a throwaway store *before* it is imported.
    os.environ["VECTOR_DB_PATH"] = str(work / "vector_db")
    os.environ.setdefault("EMBED_DEVICE", "cpu")

    print(f"\nsmoke test — {pdf.name} (first {args.pages} pages)")
    print(f"{DIM}scratch: {work}   parsed output: {debug_dir}{RESET}\n")
    r = Runner(args.verbose)
    state = {}

    # ---------------------------------------------------------- stage 1
    def parse_stage():
        use_package("parse_manager")
        from layout_detector import LayoutDetector
        from pdf_parser import parse
        from txt_processor import process_layout_json

        det = LayoutDetector()
        state["provider"] = det.provider
        # parse() writes <stem>.json (the layout); the processed blocks must
        # go somewhere else or they overwrite it.
        layout_path = parse(pdf, detector=det, max_pages=args.pages, out_dir=debug_dir)
        regions = sum(len(p["regions"]) for p in json.loads(Path(layout_path).read_text())["pages"])
        blocks_path = debug_dir / f"{pdf.stem}.blocks.json"
        txt = process_layout_json(layout_path, output_txt=debug_dir / f"{pdf.stem}.txt",
                                  output_json=blocks_path, update_registry=False)
        blocks = json.loads(blocks_path.read_text())["blocks"]
        if not blocks:
            raise AssertionError("no blocks produced")
        if not any(b["type"] == "paragraph" for b in blocks):
            raise AssertionError("no paragraph blocks — parsing produced no body text")
        state["blocks_json"] = blocks_path
        state["txt"] = txt
        if args.verbose:
            print(f"       {DIM}first block: {blocks[0]['text'][:90]}{RESET}")
        return f"{regions} regions, {len(blocks)} blocks, on {det.provider.replace('ExecutionProvider','')}"

    # ---------------------------------------------------------- stage 2
    def chunk_stage():
        use_package("embedding_manager")
        from grain_growth import chunk_file

        chunks, skipped = chunk_file(state["blocks_json"])
        if not chunks:
            raise AssertionError("chunker produced nothing")
        oversize = [c for c in chunks if len(c["text"]) / 5.1 > 512]
        if oversize:
            raise AssertionError(f"{len(oversize)} chunks exceed the 512-token window")
        prose = [c for c in chunks if c["metadata"]["block_types"] == "paragraph"]
        midcut = [c for c in prose if c["body"].rstrip()[-1] not in ".!?\"')]”"]
        state["chunks"] = chunks
        if args.verbose and chunks:
            print(f"       {DIM}chunk 0: {chunks[0]['text'][:90]}{RESET}")
        return (f"{len(chunks)} chunks, {len(skipped)} skipped, "
                f"{len(midcut)}/{len(prose)} prose cut mid-sentence")

    # ---------------------------------------------------------- stage 3
    def embed_stage():
        use_package("embedding_manager")
        from embeddings import chunk_and_embed, get_collection, search

        result = chunk_and_embed(state["blocks_json"])
        if not result["success"]:
            raise AssertionError(result.get("error", "embedding failed"))
        count = get_collection().count()
        hits = search("what is this paper about?", n_results=3)
        if not hits:
            raise AssertionError("nothing retrieved from the throwaway store")
        state["embedded"] = result["chunk_count"]
        if args.verbose:
            print(f"       {DIM}top hit d={hits[0]['distance']:.3f}: "
                  f"{hits[0]['text'][:80]}{RESET}")
        return f"{result['chunk_count']} chunks embedded, {count} in store, retrieval works"

    # ---------------------------------------------------------- stage 4
    def retrieval_stage():
        if not service_up(f"{EMBEDDING_URL}/v1/health"):
            raise SkipStage("embedding service not running (run: anneal)")
        import requests
        body = requests.post(f"{EMBEDDING_URL}/v1/search",
                             json={"query": "what method does this paper propose?",
                                   "n_results": 5}, timeout=60).json()
        hits = body["results"]
        if not hits:
            raise AssertionError("live store returned no chunks — is anything ingested?")
        d = hits[0]["distance"]
        if d > 0.9:
            raise AssertionError(f"best match is unrelated (distance {d:.3f})")
        state["live_hits"] = hits
        if args.verbose:
            for h in hits[:3]:
                m = h["metadata"]
                print(f"       {DIM}d={h['distance']:.3f} {str(m.get('title'))[:44]} › "
                      f"{str(m.get('section'))[:24]}{RESET}")
        return f"{len(hits)} chunks, best distance {d:.3f}"

    # ---------------------------------------------------------- stage 5
    def answer_stage():
        if not service_up(f"{EMBEDDING_URL}/v1/health"):
            raise SkipStage("embedding service not running")
        use_package("rag_setup")
        import rag
        from common import settings as settings_store

        cfg = settings_store.load()
        backend = cfg["llm"]["backend"]
        if backend == "local" and not service_up(f"{cfg['llm']['local']['url']}/api/version"):
            raise SkipStage("ollama not running (sudo systemctl start ollama)")

        question = "What problem does this paper address?"
        context, sources, warnings = rag.retrieve(question, None, web=False, cfg=cfg)
        if not sources:
            raise AssertionError("no sources retrieved")
        answer = "".join(rag.answer_stream(context, question, cfg)).strip()
        if not answer:
            raise AssertionError("model returned an empty answer")
        cited = any(f"[{i}]" in answer for i in range(1, len(sources) + 1))
        if args.verbose:
            print(f"       {DIM}{answer[:160]}{RESET}")
        note = f"{len(answer)} chars via {backend}, {len(sources)} sources"
        return note + ("" if cited else f" {YELLOW}(no [n] citation){RESET}")

    # ---------------------------------------------------------- stage 6
    def ui_stage():
        if not service_up(f"{UI_URL}/v1/status"):
            raise SkipStage("UI service not running (run: anneal)")
        import requests
        status = requests.get(f"{UI_URL}/v1/status", timeout=5).json()
        checks = []
        for path in ("/v1/papers", "/v1/settings", "/v1/ingestion", "/v1/chats"):
            resp = requests.get(f"{UI_URL}{path}", timeout=15)
            if not resp.ok:
                raise AssertionError(f"{path} returned {resp.status_code}")
            checks.append(path)
        page = requests.get(UI_URL, timeout=10).text
        served = "React build" if "/assets/" in page else "legacy single page"
        dots = [k for k in ("ollama", "embeddings", "registry") if status.get(k)]
        return f"{len(checks)} endpoints ok, serving {served}, up: {','.join(dots) or 'none'}"

    # ---------------------------------------------------------- stage 7
    def query_stream_stage():
        if not service_up(f"{UI_URL}/v1/status"):
            raise SkipStage("UI service not running")
        import requests
        status = requests.get(f"{UI_URL}/v1/status", timeout=5).json()
        if status["backend"] == "local" and not status.get("ollama"):
            raise SkipStage("ollama not running")
        events, answer, chat_id = [], "", None
        with requests.post(f"{UI_URL}/v1/query", json={"query": "Summarise this corpus in one sentence."},
                           stream=True, timeout=300) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                events.append(ev["type"])
                if ev["type"] == "chat":
                    chat_id = ev["chat_id"]
                elif ev["type"] == "delta":
                    answer += ev["text"]
                elif ev["type"] == "error":
                    raise AssertionError(ev["message"])
        if "done" not in events:
            raise AssertionError(f"stream did not complete: {events[-3:]}")
        if not answer.strip():
            raise AssertionError("stream produced no answer text")
        # the chat must have been persisted, and be embeddable into memory
        chat = requests.get(f"{UI_URL}/v1/chats/{chat_id}", timeout=10).json()
        roles = [m["role"] for m in chat["messages"]]
        if roles[-2:] != ["user", "assistant"]:
            raise AssertionError(f"chat not persisted correctly: {roles}")
        mem = requests.post(f"{UI_URL}/v1/chats/{chat_id}/embed", timeout=120)
        memnote = f", saved {mem.json()['embedded']} pair(s) to memory" if mem.ok else ""
        requests.delete(f"{UI_URL}/v1/chats/{chat_id}", timeout=10)     # clean up after ourselves
        deltas = events.count("delta")
        return f"{deltas} stream events, {len(answer)} chars{memnote}"

    try:
        print("offline stages (no services, no project state touched)")
        ok1 = r.stage("1. parse       PDF -> blocks", parse_stage)
        ok2 = r.stage("2. chunk       blocks -> chunks", chunk_stage) if ok1 else r.stage(
            "2. chunk       blocks -> chunks", lambda: (_ for _ in ()).throw(SkipStage("stage 1 failed")))
        if ok2:
            r.stage("3. embed       chunks -> vectors", embed_stage)
        else:
            r.stage("3. embed       chunks -> vectors",
                    lambda: (_ for _ in ()).throw(SkipStage("stage 2 failed")))

        if not args.offline:
            print("\nlive stages (need the running services)")
            r.stage("4. retrieval   live vector store", retrieval_stage)
            r.stage("5. answer      retrieval + LLM", answer_stage)
            r.stage("6. UI          endpoints + build", ui_stage)
            r.stage("7. query       full UI stream + chat", query_stream_stage)
        code = r.summary()

        if any(s is SKIP for _, s, _ in r.results):
            print(f"\n{DIM}to run the skipped stages:  anneal{RESET}")
        return code
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
