"""End-to-end retrieval evaluation, round 2: 3 parsers x 3 chunkers, 9 cells.

    python evals/scripts/run_rag_eval.py                # all 9, checkpointed
    python evals/scripts/run_rag_eval.py --cells current:grain_growth
    python evals/scripts/run_rag_eval.py --no-generate  # retrieval metrics only

For each cell: chunk the cached parse, embed into an isolated vector store,
run every question through the product's own retrieval and generation path,
score retrieval, save everything, then drop the collection and move on.

**The production vector store is never touched.** Everything happens in
evals/vector_db_eval, set through VECTOR_DB_PATH before embedding_manager is
imported.

What changed from round 1
-------------------------
- Semantic chunking is gone: round 1 settled it (last on every parser at an
  equal character budget, 2.2x slower).
- No LLM judge in the loop. Qwen's answers and the ten retrieved chunks are
  saved per question, and answers are scored offline against the reference
  answer, short answer and atomic facts (string metrics and NLI).
- Every span metric is computed twice: exact match, and near match (NEAR_MATCH
  of the span's words in a span-length window). An exact match penalises
  parsers whose text engine differs from the one the spans were verified with.
- Rows checkpoint every 10 questions, so a crash costs minutes, not a cell.

Metrics
-------
Retrieval metrics are the TREC / MS MARCO / BEIR set: hit rate, precision, MRR,
nDCG. Every rank-based metric is also reported at a fixed character budget,
because a fixed k favours whichever cell has the largest chunks.

Outputs
-------
    evals/Reports/rag_results_round2.json          per-cell aggregates
    evals/Reports/rag_round2_rows/<parser>__<chunker>.json
                                                    per-question rows: metrics,
                                                    retrieved chunks, answer
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DB = REPO / "evals" / "vector_db_eval"
os.environ["VECTOR_DB_PATH"] = str(EVAL_DB)      # before embedding_manager loads

sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(REPO / "app" / "embedding_manager"))
sys.path.insert(0, str(REPO / "app" / "rag_setup"))
sys.path.insert(0, str(REPO / "evals" / "scripts"))

PARSED = REPO / "evals" / "parsed"
QDIR = REPO / "evals" / "questions"
REPORTS = REPO / "evals" / "Reports"
RESULTS = REPORTS / "rag_results_round2.json"
ROWS = REPORTS / "rag_round2_rows"

PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "grain_growth"]
KS = (1, 3, 5, 10)
BUDGETS = (2000, 4000, 8000)
TOP_K = 10                       # retrieved per query; metrics slice into it
CONTEXT_K = 6                    # what the product hands the model
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CHECKPOINT_EVERY = 10


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


# The same comparison the question set was verified with: ligatures folded,
# wrap hyphens removed, punctuation dropped.
from build_questions import text_key, write_json                  # noqa: E402
from make_figures import NEAR_MATCH, span_overlap                 # noqa: E402


# ============================================================ index
def unload_llm(cfg):
    """Evict the answering model so bulk indexing has the card to itself.

    Ollama reloads it on the next query, a few seconds once per cell, which is
    cheaper than competing for 4 GB with the embedder and failing with OOM.
    """
    import requests
    try:
        requests.post(f"{cfg['llm']['local']['url'].rstrip('/')}/api/generate",
                      json={"model": cfg["llm"]["local"]["model"], "keep_alive": 0},
                      timeout=30)
        time.sleep(3)
    except Exception:                                            # noqa: BLE001
        pass


def build_index(parser, chunker, stems, device):
    """Chunk the cached parse and embed it into a throwaway collection."""
    import chromadb
    from run_matrix import CHUNKERS as CHUNK_IMPL
    from grain_growth import FENCED_TYPES, fence
    from embeddings import BGEEmbeddingFunction

    client = chromadb.PersistentClient(path=str(EVAL_DB))
    name = f"eval_{parser}_{chunker}"
    try:
        client.delete_collection(name)
    except Exception:                                            # noqa: BLE001
        pass
    fn = BGEEmbeddingFunction(model_name=EMBED_MODEL, device=device,
                              normalize_embeddings=True)
    col = client.get_or_create_collection(name=name, embedding_function=fn,
                                          metadata={"hnsw:space": "cosine"})

    total, missing = 0, 0
    for i, stem in enumerate(stems, 1):
        path = PARSED / parser / f"{stem}.json"
        if not path.exists():
            missing += 1
            continue
        blocks = json.loads(path.read_text(encoding="utf-8"))["blocks"]
        text = "\n\n".join(
            fence(b["type"], b["text"]) if b["type"] in FENCED_TYPES else b["text"]
            for b in blocks if b.get("text"))
        doc = {"stem": stem, "blocks": blocks, "text": text}
        try:
            chunks = [c for c in CHUNK_IMPL[chunker](doc) if c.strip()]
        except Exception as e:                                   # noqa: BLE001
            print(f"    chunk failed {stem}: {str(e)[:70]}", flush=True)
            continue
        if not chunks:
            continue
        col.add(documents=chunks,
                metadatas=[{"filename": stem, "chunk_id": j} for j in range(len(chunks))],
                ids=[f"{stem}__{j}" for j in range(len(chunks))])
        total += len(chunks)
        if i % 50 == 0:
            print(f"    indexed {i}/{len(stems)} papers, {total} chunks", flush=True)
    if missing:
        print(f"    WARNING: {missing} papers have no cached {parser} parse", flush=True)
    return col, total


def drop_index(parser, chunker):
    import chromadb
    client = chromadb.PersistentClient(path=str(EVAL_DB))
    try:
        client.delete_collection(f"eval_{parser}_{chunker}")
    except Exception:                                            # noqa: BLE001
        pass


# ============================================================ metrics
def ndcg(gains, ideal):
    def dcg(xs):
        return sum(g / math.log2(i + 2) for i, g in enumerate(xs))
    denom = dcg(sorted(ideal, reverse=True))
    return dcg(gains) / denom if denom else 0.0


def score_question(q, hits):
    """hits: ordered list of (filename, chunk_text). Graded relevance:
    2 = chunk carries the answer span, 1 = right paper, 0 = neither.

    Span metrics come in two forms: exact (substring in text_key form) and
    `_near` (NEAR_MATCH of the span's words in one window)."""
    span = text_key(q["answer_span"])
    kws = [norm(k) for k in q.get("keywords", []) if k.strip()]
    paper = q["paper"]
    keyed = [(fn_, text, text_key(text)) for fn_, text in hits]

    out = {}
    paper_ranks = [i + 1 for i, (fn_, _, _) in enumerate(keyed) if fn_ == paper]
    for k in KS:
        out[f"paper_hit@{k}"] = 1.0 if any(r <= k for r in paper_ranks) else 0.0
    out["mrr_paper"] = 1.0 / paper_ranks[0] if paper_ranks else 0.0

    for suffix, has in (("", lambda t: span in t),
                        ("_near", lambda t: span_overlap(span, t) >= NEAR_MATCH)):
        found = [has(t) for _, _, t in keyed]
        grades = [2 if f else (1 if fn_ == paper else 0) for f, (fn_, _, _) in zip(found, keyed)]
        span_ranks = [i + 1 for i, f in enumerate(found) if f]
        for k in KS:
            out[f"span_hit{suffix}@{k}"] = 1.0 if any(r <= k for r in span_ranks) else 0.0
            out[f"precision{suffix}@{k}"] = sum(1 for g in grades[:k] if g > 0) / k
        out[f"mrr{suffix}"] = 1.0 / span_ranks[0] if span_ranks else 0.0
        # one span per question, so the ideal ranking is that chunk first
        out[f"ndcg{suffix}@10"] = ndcg(grades[:10], [2] + [1] * 9)
        # budget-fair: fill a character budget in rank order, then ask the same thing
        for b in BUDGETS:
            used, hit = 0, False
            for f, (_, text, _) in zip(found, keyed):
                if used >= b:
                    break
                used += len(text)
                if f:
                    hit = True
                    break
            out[f"span_hit{suffix}@{b}ch"] = 1.0 if hit else 0.0

    ctx = norm(" ".join(t for _, t in hits[:CONTEXT_K]))
    out["keyword_recall"] = (sum(1 for k in kws if k in ctx) / len(kws)) if kws else None
    out["context_chars"] = sum(len(t) for _, t in hits[:CONTEXT_K])
    return out


# ============================================================ run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", help="parser:chunker, default all 9")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--limit-questions", type=int)
    args = ap.parse_args()

    import rag
    from common.settings import load
    from sentence_transformers import SentenceTransformer

    cfg = load()
    REPORTS.mkdir(parents=True, exist_ok=True)
    ROWS.mkdir(parents=True, exist_ok=True)
    data = json.loads((QDIR / "dataset.json").read_text())
    questions = data["questions"][:args.limit_questions] if args.limit_questions \
        else data["questions"]
    # index the whole corpus, not just the papers questions came from
    corpus = json.loads((REPO / "evals" / "corpus" / "manifest.json").read_text())
    stems = sorted({p["arxiv_id"] for p in corpus["papers"]})

    cells = [tuple(c.split(":")) for c in args.cells] if args.cells else \
        [(p, c) for p in PARSERS for c in CHUNKERS]

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {
        "config": {"top_k": TOP_K, "context_k": CONTEXT_K, "ks": list(KS),
                   "budgets": list(BUDGETS), "embed_model": EMBED_MODEL,
                   "near_match": NEAR_MATCH, "llm": cfg["llm"]["local"]["model"],
                   "questions": len(questions), "corpus_papers": len(stems)},
        "cells": {}}

    sim_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    ideal_vecs = sim_model.encode([q["ideal_answer"] for q in questions],
                                  normalize_embeddings=True, show_progress_bar=False)

    for parser, chunker in cells:
        key = f"{parser}|{chunker}"
        if results["cells"].get(key, {}).get("complete"):
            print(f"== {key}: already done, skipping")
            continue
        rows_path = ROWS / f"{parser}__{chunker}.json"
        rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
        done = {r["qid"] for r in rows if "error" not in r and (args.no_generate or "answer" in r)}
        rows = [r for r in rows if r["qid"] in done]
        print(f"\n== {key}" + (f" (resuming: {len(done)} questions already done)" if done else ""),
              flush=True)

        t0 = time.time()
        unload_llm(cfg)
        try:
            col, n_chunks = build_index(parser, chunker, stems, args.device)
        except Exception as e:                                   # noqa: BLE001
            results["cells"][key] = {"error": f"index: {str(e)[:200]}", "complete": False}
            write_json(RESULTS, results, indent=1)
            print(f"   INDEX FAILED: {str(e)[:120]}")
            continue
        index_seconds = time.time() - t0
        print(f"   {n_chunks} chunks indexed in {index_seconds:.0f}s", flush=True)

        # Re-open with a CPU embedder for querying. One query embeds in
        # milliseconds on CPU, and it leaves all 4 GB for the answering model.
        import chromadb as _ch
        from embeddings import BGEEmbeddingFunction as _BGE
        col = _ch.PersistentClient(path=str(EVAL_DB)).get_collection(
            f"eval_{parser}_{chunker}",
            embedding_function=_BGE(model_name=EMBED_MODEL, device="cpu",
                                    normalize_embeddings=True))

        t_q = time.time()
        for qi, q in enumerate(questions):
            if q["qid"] in done:
                continue
            try:
                res = col.query(query_texts=[q["question"]], n_results=TOP_K,
                                include=["documents", "metadatas"])
                hits = list(zip([m["filename"] for m in res["metadatas"][0]],
                                res["documents"][0]))
            except Exception as e:                               # noqa: BLE001
                rows.append({"qid": q["qid"], "error": str(e)[:150]})
                continue
            row = {"qid": q["qid"], "paper": q["paper"], "source": q.get("source"),
                   **score_question(q, hits), "hits": [[f, t] for f, t in hits]}

            if not args.no_generate:
                results_fmt = [{"text": t, "metadata": {"filename": f}}
                               for f, t in hits[:CONTEXT_K]]
                context, _srcs = rag.build_context(results_fmt)
                try:
                    ans = rag.answer(context, q["question"], cfg)
                except Exception as e:                           # noqa: BLE001
                    ans = ""
                    row["gen_error"] = str(e)[:120]
                row["answer"] = ans[:4000]
                if ans:
                    v = sim_model.encode([ans], normalize_embeddings=True,
                                         show_progress_bar=False)[0]
                    row["answer_similarity"] = float(v @ ideal_vecs[qi])
                    kws = [norm(k) for k in q.get("keywords", [])]
                    row["answer_keyword_recall"] = (
                        sum(1 for k in kws if k in norm(ans)) / len(kws)) if kws else None
            rows.append(row)
            new = len(rows) - len(done)
            if new % CHECKPOINT_EVERY == 0:
                write_json(rows_path, rows)
            if new % 25 == 0:
                rate = (time.time() - t_q) / new
                left = (len(questions) - len(rows)) * rate / 60
                print(f"   {len(rows)}/{len(questions)} questions "
                      f"({rate:.0f}s each, ~{left:.0f} min left in cell)", flush=True)
        write_json(rows_path, rows)

        # aggregate
        def mean(field, sub=None):
            vals = [r[field] for r in (sub or rows) if isinstance(r.get(field), (int, float))]
            return sum(vals) / len(vals) if vals else None

        fields = [f for f in rows[0] if isinstance(rows[0].get(f), (int, float))] if rows else []
        agg = {f: mean(f) for f in fields}
        agg.update({"chunks": n_chunks, "index_seconds": round(index_seconds, 1),
                    "seconds": round(time.time() - t0, 1), "questions": len(rows),
                    "errors": sum(1 for r in rows if "error" in r or "gen_error" in r),
                    "complete": True})
        # do never-trained YOLO-folder papers behave differently from fresh arXiv ones?
        agg["by_source"] = {}
        for src in sorted({r.get("source") for r in rows if r.get("source")}):
            sub = [r for r in rows if r.get("source") == src]
            agg["by_source"][src] = {"n": len(sub), "span_hit_near@5": mean("span_hit_near@5", sub),
                                     "span_hit_near@4000ch": mean("span_hit_near@4000ch", sub)}
        results["cells"][key] = agg
        write_json(RESULTS, results, indent=1)
        drop_index(parser, chunker)
        print(f"   span_hit@5={agg.get('span_hit@5', 0):.3f} near={agg.get('span_hit_near@5', 0):.3f} "
              f"@4000ch near={agg.get('span_hit_near@4000ch', 0):.3f} ({agg['seconds']/60:.0f} min)",
              flush=True)

    if EVAL_DB.exists():
        shutil.rmtree(EVAL_DB, ignore_errors=True)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    q = {"answer_span": "the model reaches 43 percent lower latency on CPU", "keywords": ["latency"],
         "paper": "p1"}
    s = score_question(q, [("p2", "unrelated text"), ("p1", "The model reaches 43 percent lower latency on CPU.")])
    assert s["span_hit@1"] == 0.0 and s["span_hit@3"] == 1.0 and s["mrr"] == 0.5
    # one word of twelve differs ("percent" vs "%"): 11/12 is above NEAR_MATCH, not exact
    q = {**q, "answer_span": "on the benchmark the model reaches 43 percent lower latency on CPU"}
    s = score_question(q, [("p1", "On the benchmark, the model reaches 43 % lower latency on CPU.")])
    assert s["span_hit@1"] == 0.0 and s["span_hit_near@1"] == 1.0 and s["mrr_near"] == 1.0
    main()
