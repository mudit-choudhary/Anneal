"""End-to-end retrieval evaluation: 3 parsers x 4 chunkers, 12 cells.

    python evals/scripts/run_rag_eval.py                # all 12, checkpointed
    python evals/scripts/run_rag_eval.py --cells current:grain_growth
    python evals/scripts/run_rag_eval.py --no-generate  # retrieval metrics only

For each cell: chunk the cached parse, embed into an isolated vector store,
run every question through the product's own retrieval and generation path,
score, then drop the collection and move on.

**The production vector store is never touched.** Everything happens in
evals/vector_db_eval, set through VECTOR_DB_PATH before embedding_manager is
imported. "Clear the vector DB between combinations" means dropping that
collection, not the live one.

Metrics, and why these
----------------------
Retrieval metrics are the TREC / MS MARCO / BEIR set — hit rate, recall,
precision, MRR, nDCG — because those are what the IR literature is built on.
nDCG is reported with graded relevance and is the one to weight most heavily:
it accounts for rank position and correlates best with end-to-end RAG quality.

Every rank-based metric is reported twice: at fixed k, which is conventional
but favours whichever cell has the largest chunks, and at a fixed character
budget, which is comparable across cells. The budget figures are the ones to
trust when comparing chunkers.

Generation metrics follow RAGAS: faithfulness (is the answer grounded in the
retrieved context), answer correctness against a written ideal answer, and
context sufficiency. Deterministic proxies (keyword coverage, embedding
similarity) are recorded alongside because LLM judges are noisy.
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

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "embedding_manager"))
sys.path.insert(0, str(REPO / "rag_setup"))
sys.path.insert(0, str(REPO / "evals" / "scripts"))

PARSED = REPO / "evals" / "parsed"
QDIR = REPO / "evals" / "questions"
REPORTS = REPO / "evals" / "Reports"
RESULTS = REPORTS / "rag_results.json"

PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "semantic", "grain_growth"]
KS = (1, 3, 5, 10)
BUDGETS = (2000, 4000, 8000)
TOP_K = 10                       # retrieved per query; metrics slice into it
CONTEXT_K = 6                    # what the product hands the model
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


# The same comparison the question set was verified with: ligatures folded,
# wrap hyphens removed, punctuation dropped. Scoring with anything stricter
# would count a chunk that plainly contains the span as a miss.
from build_questions import text_key                              # noqa: E402


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
    from chunking import FENCED_TYPES, fence
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

    total = 0
    for i, stem in enumerate(stems, 1):
        path = PARSED / parser / f"{stem}.json"
        if not path.exists():
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
        if i % 40 == 0:
            print(f"    indexed {i}/{len(stems)} papers, {total} chunks", flush=True)
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
    2 = chunk carries the verbatim answer span, 1 = right paper, 0 = neither."""
    span = text_key(q["answer_span"])
    kws = [norm(k) for k in q.get("keywords", []) if k.strip()]
    paper = q["paper"]

    grades, span_ranks, paper_ranks, sizes = [], [], [], []
    for i, (fn_, text) in enumerate(hits):
        t = text_key(text)
        has_span = span in t
        right = fn_ == paper
        grades.append(2 if has_span else (1 if right else 0))
        if has_span:
            span_ranks.append(i + 1)
        if right:
            paper_ranks.append(i + 1)
        sizes.append(len(text))

    out = {}
    for k in KS:
        out[f"span_hit@{k}"] = 1.0 if any(r <= k for r in span_ranks) else 0.0
        out[f"paper_hit@{k}"] = 1.0 if any(r <= k for r in paper_ranks) else 0.0
        out[f"precision@{k}"] = sum(1 for g in grades[:k] if g > 0) / k
    out["mrr"] = 1.0 / span_ranks[0] if span_ranks else 0.0
    out["mrr_paper"] = 1.0 / paper_ranks[0] if paper_ranks else 0.0
    # one span per question, so the ideal ranking is that chunk first
    out["ndcg@10"] = ndcg(grades[:10], [2] + [1] * 9)

    # budget-fair: fill a character budget in rank order, then ask the same thing
    for b in BUDGETS:
        used, found = 0, False
        for (fn_, text) in hits:
            if used >= b:
                break
            used += len(text)
            if span in text_key(text):
                found = True
                break
        out[f"span_hit@{b}ch"] = 1.0 if found else 0.0

    ctx = norm(" ".join(t for _, t in hits[:CONTEXT_K]))
    out["keyword_recall"] = (sum(1 for k in kws if k in ctx) / len(kws)) if kws else None
    out["context_chars"] = sum(len(t) for _, t in hits[:CONTEXT_K])
    return out


JUDGE = """You are grading a retrieval-augmented answer. Be strict.

QUESTION: {q}

REFERENCE ANSWER (correct by construction): {ideal}

RETRIEVED CONTEXT GIVEN TO THE MODEL:
{ctx}

THE MODEL'S ANSWER: {ans}

Score three things, each 0 to 2:
- "faithfulness": 2 if every claim in the model's answer is supported by the
  context, 1 if mostly, 0 if it asserts things the context does not contain.
- "correctness": 2 if the model's answer matches the reference, 1 if partly,
  0 if wrong or evasive.
- "context_sufficiency": 2 if the context alone contains what is needed to
  answer, 1 if partly, 0 if not.

Reply with JSON only: {{"faithfulness": n, "correctness": n, "context_sufficiency": n}}
"""


def judge(q, context, answer, cfg):
    import requests
    local = cfg["llm"]["local"]
    prompt = JUDGE.format(q=q["question"], ideal=q["ideal_answer"],
                          ctx=context[:6000], ans=answer[:2000])
    try:
        r = requests.post(f"{local['url'].rstrip('/')}/api/chat", json={
            "model": local["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False,
            "options": {"num_ctx": int(local.get("num_ctx", 6144)), "temperature": 0.0},
        }, timeout=300)
        r.raise_for_status()
        m = re.search(r"\{.*\}", r.json()["message"]["content"], re.S)
        if not m:
            return {}
        d = json.loads(m.group(0))
        return {k: float(d[k]) / 2.0 for k in
                ("faithfulness", "correctness", "context_sufficiency") if k in d}
    except Exception:                                            # noqa: BLE001
        return {}


# ============================================================ run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", help="parser:chunker, default all 12")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--limit-questions", type=int)
    args = ap.parse_args()

    # The semantic chunker stays on CPU. Putting it on the card alongside the
    # indexing embedder exhausted 4 GB and failed a third of the corpus with
    # CUDA OOM; the time it saves is not worth the fragility.
    os.environ.setdefault("SEMANTIC_DEVICE", "cpu")

    import rag
    from common.settings import load
    from sentence_transformers import SentenceTransformer

    cfg = load()
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = json.loads((QDIR / "dataset.json").read_text())
    questions = data["questions"][:args.limit_questions] if args.limit_questions \
        else data["questions"]
    stems = sorted({q["paper"] for q in
                    json.loads((QDIR / "candidates.json").read_text())["questions"]}
                   | {q["paper"] for q in questions})
    # index the whole corpus, not just the papers questions came from
    corpus = json.loads((REPO / "evals" / "corpus" / "manifest.json").read_text())
    stems = sorted({p["arxiv_id"] for p in corpus["papers"]})

    cells = [tuple(c.split(":")) for c in args.cells] if args.cells else \
        [(p, c) for p in PARSERS for c in CHUNKERS]

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {
        "config": {"top_k": TOP_K, "context_k": CONTEXT_K, "ks": list(KS),
                   "budgets": list(BUDGETS), "embed_model": EMBED_MODEL,
                   "llm": cfg["llm"]["local"]["model"],
                   "questions": len(questions), "corpus_papers": len(stems)},
        "cells": {}}

    sim_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    ideal_vecs = sim_model.encode([q["ideal_answer"] for q in questions],
                                  normalize_embeddings=True, show_progress_bar=False)

    for parser, chunker in cells:
        key = f"{parser}|{chunker}"
        if key in results["cells"] and results["cells"][key].get("complete"):
            print(f"== {key}: already done, skipping")
            continue
        print(f"\n== {key}", flush=True)
        t0 = time.time()
        unload_llm(cfg)
        try:
            col, n_chunks = build_index(parser, chunker, stems, args.device)
        except Exception as e:                                   # noqa: BLE001
            results["cells"][key] = {"error": f"index: {str(e)[:200]}", "complete": False}
            RESULTS.write_text(json.dumps(results, indent=1))
            print(f"   INDEX FAILED: {str(e)[:120]}")
            continue
        print(f"   {n_chunks} chunks indexed in {time.time()-t0:.0f}s", flush=True)

        # Re-open with a CPU embedder for querying. One query embeds in
        # milliseconds on CPU, and it leaves all 4 GB for the answering model.
        if not args.no_generate:
            import chromadb as _ch
            from embeddings import BGEEmbeddingFunction as _BGE
            col = _ch.PersistentClient(path=str(EVAL_DB)).get_collection(
                f"eval_{parser}_{chunker}",
                embedding_function=_BGE(model_name=EMBED_MODEL, device="cpu",
                                        normalize_embeddings=True))

        rows = []
        for qi, q in enumerate(questions):
            try:
                res = col.query(query_texts=[q["question"]], n_results=TOP_K,
                                include=["documents", "metadatas"])
                hits = list(zip([m["filename"] for m in res["metadatas"][0]],
                                res["documents"][0]))
            except Exception as e:                               # noqa: BLE001
                rows.append({"qid": q["qid"], "error": str(e)[:150]})
                continue
            row = {"qid": q["qid"], "paper": q["paper"],
                   "generated_from": q["generated_from"], **score_question(q, hits)}

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
                    row.update(judge(q, context, ans, cfg))
            rows.append(row)
            if (qi + 1) % 20 == 0:
                print(f"   {qi+1}/{len(questions)} questions "
                      f"({(time.time()-t0)/60:.0f} min)", flush=True)

        # aggregate
        def mean(field):
            vals = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
            return sum(vals) / len(vals) if vals else None

        fields = ([f"span_hit@{k}" for k in KS] + [f"paper_hit@{k}" for k in KS]
                  + [f"precision@{k}" for k in KS] + [f"span_hit@{b}ch" for b in BUDGETS]
                  + ["mrr", "mrr_paper", "ndcg@10", "keyword_recall", "context_chars",
                     "answer_similarity", "answer_keyword_recall",
                     "faithfulness", "correctness", "context_sufficiency"])
        agg = {f: mean(f) for f in fields}
        agg.update({"chunks": n_chunks, "seconds": round(time.time() - t0, 1),
                    "questions": len(rows), "complete": True})
        # bias check: does an arm do better on questions written from its own text?
        agg["by_source"] = {}
        for src in PARSERS:
            sub = [r for r in rows if r.get("generated_from") == src]
            if sub:
                agg["by_source"][src] = {
                    "n": len(sub),
                    "span_hit@5": sum(r.get("span_hit@5", 0) for r in sub) / len(sub),
                    "ndcg@10": sum(r.get("ndcg@10", 0) for r in sub) / len(sub)}
        results["cells"][key] = agg
        results["cells"][key]["_rows"] = rows
        RESULTS.write_text(json.dumps(results, indent=1))
        drop_index(parser, chunker)
        print(f"   span_hit@5={agg['span_hit@5']:.3f} ndcg@10={agg['ndcg@10']:.3f} "
              f"correctness={agg.get('correctness')} ({agg['seconds']/60:.0f} min)",
              flush=True)

    if EVAL_DB.exists():
        shutil.rmtree(EVAL_DB, ignore_errors=True)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
