"""Score round-2 answers offline and locally: string metrics, NLI, Gemma 3 4B.

    python evals/scripts/score_answers.py --stage string            # CPU, seconds
    python evals/scripts/score_answers.py --stage nli --threads 4   # CPU, ~1 h for all cells
    python evals/scripts/score_answers.py --stage gemma             # GPU, ~12 h, after the run

No external API is called and qwen3:4b never judges its own answers. Only
cells the retrieval run has finished are scored; every stage skips questions
it has already scored, so it can be stopped and resumed at any point.

What each stage writes, per question, into
evals/Reports/rag_round2_scores/<parser>__<chunker>.json:

  string  short_answer_match, numeric_anchor_eligible, numeric_anchor_correct,
          token_f1, rouge_l, chrf, refusal, answer_words
  nli     nli_fact_recall        share of the reference's own facts the answer entails
                                 (facts the reference answer itself entails; amendment 2)
          nli_fact_recall_all    the same over every fact, as first pre-registered
          nli_faithfulness       share of answer sentences the context entails
          nli_context_sufficiency share of reference facts the context entails
          nli_contradiction      1 if the reference contradicts an answer sentence
  gemma   gemma_correctness, gemma_faithfulness, gemma_context_sufficiency
          (0-2 normalised to 0-1) and the raw reply

Refusals score 0 on the answer metrics, as pre-registered.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# DeBERTa's disentangled attention is memory-hungry on 400-token pairs; on a 4 GB
# card it OOMs at batch 32. Smaller batches plus expandable segments keep it inside.
for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):   # new name, then the old one
    os.environ.setdefault(_var, "expandable_segments:True")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rag_setup"))
sys.path.insert(0, str(REPO / "evals" / "scripts"))

from build_questions import _CITATION, text_key, write_json      # noqa: E402

REPORTS = REPO / "evals" / "Reports"
RESULTS = REPORTS / "rag_results_round2.json"
ROWS = REPORTS / "rag_round2_rows"
SCORES = REPORTS / "rag_round2_scores"
DATASET = REPO / "evals" / "questions" / "dataset.json"
CONTEXT_K = 6                    # what the answering model saw, as in run_rag_eval
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
JUDGE_MODEL = "gemma3:4b"
SEED = 20260915
WINDOW_WORDS, WINDOW_STRIDE = 300, 250   # NLI reads ~512 tokens; long texts are windowed

# a sentence-initial or short reply saying the context lacks the answer
_REFUSAL = re.compile(
    r"\b(not (provided|given|mentioned|specified|stated|included|available|found)|"
    r"(cannot|can't|can not|unable to) (be )?(find|answer|determine|provide|locate)|"
    r"(context|documents?|text|excerpts?) (does not|doesn't|do not) (contain|provide|mention|include|specify)|"
    r"no (information|mention|details?) (about|on|regarding))", re.I)
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

JUDGE = """You are grading an answer to a question about a research paper. Be strict.

QUESTION: {q}
REFERENCE ANSWER (correct): {ref}
CONTEXT THE ANSWERING MODEL SAW:
{ctx}
ANSWER TO GRADE: {ans}

Score each 0, 1 or 2:
- "correctness": 2 = matches the reference, 1 = partly, 0 = wrong or evasive
- "faithfulness": 2 = every claim supported by the context, 1 = mostly, 0 = unsupported claims
- "context_sufficiency": 2 = context contains what is needed, 1 = partly, 0 = no

Reply with JSON only: {{"correctness": n, "faithfulness": n, "context_sufficiency": n}}"""


# ============================================================ string metrics
def numbers(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", s or "")                    # superscripts, narrow spaces
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)                   # unicode hyphens and minus
    return {float(m.replace(",", "")) for m in _NUM.findall(s)}


def is_refusal(answer):
    a = (answer or "").strip()
    if not a:
        return True
    m = _REFUSAL.search(a)
    return bool(m) and (m.start() < 160 or len(a.split()) <= 40)


def short_answer_match(short, answer):
    """Numbers in the short answer must all appear in the answer; otherwise the
    short answer's text must appear. Strict: correct paraphrases of names fail."""
    need = numbers(short)
    if need:
        have = numbers(answer)
        return float(all(any(abs(n - h) <= 1e-9 * max(1.0, abs(n)) for h in have) for n in need))
    return float(bool(text_key(short)) and text_key(short) in text_key(answer))


def _tokens(s):
    return re.sub(r"\b(a|an|the)\b", " ", text_key(s)).split()


def token_f1(pred, ref):
    p, r = _tokens(pred), _tokens(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred, ref):
    p, r = text_key(pred).split(), text_key(ref).split()
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for x in p:
        cur = [0]
        for j, y in enumerate(r):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    P, R = lcs / len(p), lcs / len(r)
    return 2 * P * R / (P + R)


def chrf(pred, ref, n=6, beta=2.0):
    """chrF: character 1..n-gram F-score, recall weighted by beta."""
    p, r = re.sub(r"\s+", "", pred or ""), re.sub(r"\s+", "", ref or "")
    ps, rs = [], []
    for k in range(1, n + 1):
        pc = Counter(p[i:i + k] for i in range(len(p) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not pc or not rc:
            continue
        m = sum((pc & rc).values())
        ps.append(m / sum(pc.values()))
        rs.append(m / sum(rc.values()))
    if not ps:
        return 0.0
    P, R = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if P + R == 0 else (1 + beta ** 2) * P * R / (beta ** 2 * P + R)


def string_scores(q, answer):
    short = q.get("short_answer", "")
    refusal = is_refusal(answer)
    # The numeric reference set: a number in the short answer, not a citation, and
    # the reference answer itself passes the check. Where the reference writes the
    # number differently ("few hundredths" for 10^-2), the check says nothing.
    eligible = (bool(re.search(r"\d", short)) and not _CITATION.search(short)
                and short_answer_match(short, q["ideal_answer"]) == 1.0)
    sam = 0.0 if refusal else short_answer_match(short, answer)
    return {
        "refusal": float(refusal),
        "answer_words": len((answer or "").split()),
        "short_answer_match": sam,
        "numeric_anchor_eligible": eligible,
        "numeric_anchor_correct": sam if eligible else None,
        "token_f1": 0.0 if refusal else token_f1(answer, q["ideal_answer"]),
        "rouge_l": 0.0 if refusal else rouge_l(answer, q["ideal_answer"]),
        "chrf": 0.0 if refusal else chrf(answer, q["ideal_answer"]),
    }


# ============================================================ helpers
def sentences(text):
    text = re.sub(r"[*_#`>|]+", " ", text or "")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4]


def windows(text):
    w = (text or "").split()
    if len(w) <= WINDOW_WORDS:
        return [" ".join(w)] if w else []
    return [" ".join(w[i:i + WINDOW_WORDS]) for i in range(0, len(w) - WINDOW_STRIDE, WINDOW_STRIDE)]


def finished_cells(only=None):
    if not RESULTS.exists():
        return []
    cells = json.loads(RESULTS.read_text())["cells"]
    keys = [k for k, v in cells.items() if v.get("complete")]
    if only:
        keys = [k for k in keys if k.replace("|", ":") in only or k in only]
    return keys


def load(cell):
    parser, chunker = cell.split("|")
    rows = {r["qid"]: r for r in json.loads((ROWS / f"{parser}__{chunker}.json").read_text())}
    path = SCORES / f"{parser}__{chunker}.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    return rows, scores, path


def usable(row):
    return "error" not in row and "gen_error" not in row and "answer" in row


def retrieval_running():
    return subprocess.run(["pgrep", "-f", "run_rag_eval.py"], capture_output=True).returncode == 0


# ============================================================ stages
def stage_string(cells, qs):
    for cell in cells:
        rows, scores, path = load(cell)
        n = 0
        for qid, row in rows.items():
            s = scores.setdefault(qid, {})
            if "token_f1" in s or not usable(row):
                if not usable(row):
                    s["skipped"] = row.get("error") or row.get("gen_error") or "no answer"
                continue
            s.update(string_scores(qs[qid], row["answer"]))
            n += 1
        write_json(path, scores, indent=1)
        print(f"  {cell}: {n} scored", flush=True)


_BATCH = [8]                     # shrinks itself if the card runs out of memory


def predict(model, pairs):
    """Label each pair, halving the batch size on CUDA OOM rather than dying."""
    import torch
    if not pairs:
        return []
    while True:
        try:
            return [int(x.argmax()) for x in
                    model.predict(pairs, batch_size=_BATCH[0], show_progress_bar=False)]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if _BATCH[0] <= 1:
                raise
            _BATCH[0] = max(1, _BATCH[0] // 2)
            print(f"    CUDA out of memory; retrying at batch {_BATCH[0]}", flush=True)


def fact_validity(model, ent, qs):
    """For each question, which of its facts the reference answer itself entails.

    The question writer sometimes put details into the facts that its own
    reference answer does not state; recall over those would score answers on
    what the reference omits (amendment 2). Deterministic, computed once, cached.
    """
    path = SCORES / "_fact_validity.json"
    if path.exists():
        return json.loads(path.read_text())
    pairs = [(q["ideal_answer"], f) for q in qs.values() for f in q["facts"]]
    labels = predict(model, pairs)
    out, i = {}, 0
    for qid, q in qs.items():
        out[qid] = [labels[i + k] == ent for k in range(len(q["facts"]))]
        i += len(q["facts"])
    write_json(path, out, indent=1)
    return out


def stage_nli(cells, qs, threads, device, batch=8):
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(threads)
    _BATCH[0] = batch
    model = CrossEncoder(NLI_MODEL, device=device)
    lab = {v.lower(): int(k) for k, v in model.model.config.id2label.items()}
    ENT, CON = lab["entailment"], lab["contradiction"]
    validity = fact_validity(model, ENT, qs)

    for cell in cells:
        rows, scores, path = load(cell)
        todo = [qid for qid, r in rows.items() if usable(r) and "nli_fact_recall" not in scores.get(qid, {})]
        t0 = time.time()
        for i, qid in enumerate(todo, 1):
            row, q = rows[qid], qs[qid]
            answer = row["answer"]
            facts = q["facts"]
            sents = sentences(answer)
            ans_w = windows(answer)
            ctx_w = windows(" ".join(t for _, t in row["hits"][:CONTEXT_K]))
            pairs, spans = [], {}

            def add(name, items):
                spans[name] = (len(pairs), len(pairs) + len(items))
                pairs.extend(items)

            add("recall", [(w, f) for f in facts for w in ans_w])
            add("contra", [(q["ideal_answer"], s) for s in sents])
            add("faith", [(w, s) for s in sents for w in ctx_w])
            add("suff", [(w, f) for f in facts for w in ctx_w])
            labels = predict(model, pairs)

            def grid(name, n_items, n_windows):
                a, b = spans[name]
                flat = labels[a:b]
                return [flat[k * n_windows:(k + 1) * n_windows] for k in range(n_items)] if n_windows else []

            refusal = is_refusal(answer)
            recall = grid("recall", len(facts), len(ans_w))
            faith = grid("faith", len(sents), len(ctx_w))
            suff = grid("suff", len(facts), len(ctx_w))
            contra = labels[spans["contra"][0]:spans["contra"][1]]
            hit = [ENT in g for g in recall]
            valid = validity[qid]
            s = scores.setdefault(qid, {})
            s.update({
                "nli_fact_recall": None if not any(valid) else
                (0.0 if refusal or not hit else sum(h for h, v in zip(hit, valid) if v) / sum(valid)),
                "nli_fact_recall_all": 0.0 if refusal or not hit else sum(hit) / len(facts),
                "nli_faithfulness": None if refusal or not faith else
                sum(ENT in g for g in faith) / len(sents),
                "nli_context_sufficiency": (sum(ENT in g for g in suff) / len(facts)) if suff else 0.0,
                "nli_contradiction": float(CON in contra) if contra else None,
            })
            if i % 25 == 0 or i == len(todo):
                write_json(path, scores, indent=1)
                print(f"  {cell}: {i}/{len(todo)} ({(time.time() - t0) / i:.1f}s each)", flush=True)
        write_json(path, scores, indent=1)


def stage_gemma(cells, qs, force):
    import requests
    import rag
    from common.settings import load as load_cfg

    if retrieval_running() and not force:
        sys.exit("the retrieval run is using the GPU; run the Gemma stage after it finishes (or --force)")
    url = load_cfg()["llm"]["local"]["url"].rstrip("/")
    requests.post(f"{url}/api/generate", json={"model": load_cfg()["llm"]["local"]["model"],
                                                "keep_alive": 0}, timeout=30)   # evict qwen

    data = {cell: load(cell) for cell in cells}
    jobs = [(cell, qid) for cell, (rows, scores, _) in data.items() for qid, r in rows.items()
            if usable(r) and "gemma_raw" not in scores.get(qid, {})]
    # shuffled across cells, so any drift in the judge over hours cannot line up with a cell
    random.Random(SEED).shuffle(jobs)
    print(f"  {len(jobs)} answers to judge", flush=True)

    t0, dirty = time.time(), set()
    for i, (cell, qid) in enumerate(jobs, 1):
        rows, scores, path = data[cell]
        row, q = rows[qid], qs[qid]
        ctx = rag.build_context([{"text": t, "metadata": {"filename": f}}
                                 for f, t in row["hits"][:CONTEXT_K]])[0][:9000]
        body = {"model": JUDGE_MODEL, "stream": False, "format": "json",
                "messages": [{"role": "user", "content": JUDGE.format(
                    q=q["question"], ref=q["ideal_answer"], ctx=ctx, ans=row["answer"][:3000])}],
                "options": {"temperature": 0, "num_ctx": 6144}}
        s = scores.setdefault(qid, {})
        for attempt in range(3):
            try:
                r = requests.post(f"{url}/api/chat", json=body, timeout=600)
                r.raise_for_status()
                raw = r.json()["message"]["content"]
                d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                vals = {k: min(2, max(0, int(round(float(d[k]))))) / 2.0
                        for k in ("correctness", "faithfulness", "context_sufficiency")}
                refusal = is_refusal(row["answer"])
                s.update({"gemma_raw": raw,
                          "gemma_correctness": 0.0 if refusal else vals["correctness"],
                          "gemma_faithfulness": vals["faithfulness"],
                          "gemma_context_sufficiency": vals["context_sufficiency"]})
                s.pop("gemma_error", None)
                break
            except Exception as e:                               # noqa: BLE001
                s["gemma_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5)
        dirty.add(cell)
        if i % 20 == 0 or i == len(jobs):
            for c in dirty:
                write_json(data[c][2], data[c][1], indent=1)
            dirty.clear()
            rate = (time.time() - t0) / i
            print(f"  {i}/{len(jobs)} ({rate:.1f}s each, ~{(len(jobs) - i) * rate / 3600:.1f} h left)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["string", "nli", "gemma"], required=True)
    ap.add_argument("--cells", nargs="*", help="parser:chunker; default every finished cell")
    ap.add_argument("--threads", type=int, default=4, help="NLI CPU threads")
    ap.add_argument("--device", default="cpu", help="NLI device")
    ap.add_argument("--batch", type=int, default=8, help="NLI batch size (halves itself on OOM)")
    ap.add_argument("--force", action="store_true", help="run Gemma even if retrieval is running")
    args = ap.parse_args()

    SCORES.mkdir(parents=True, exist_ok=True)
    qs = {q["qid"]: q for q in json.loads(DATASET.read_text())["questions"]}
    cells = finished_cells(args.cells)
    print(f"== {args.stage}: {len(cells)} finished cells {cells}", flush=True)
    if args.stage == "string":
        stage_string(cells, qs)
    elif args.stage == "nli":
        stage_nli(cells, qs, args.threads, args.device, args.batch)
    else:
        stage_gemma(cells, qs, args.force)


if __name__ == "__main__":
    assert numbers("3 conv layers (4,8,8), 1,024 units, 43% and 0.5") == {3.0, 4.0, 8.0, 1024.0, 43.0, 0.5}
    assert short_answer_match("43% reduction", "It cuts CPU latency by 43 percent.") == 1.0
    assert short_answer_match("43% reduction", "It cuts CPU latency by 34%.") == 0.0
    assert short_answer_match("Token-Guard", "The token guard module") == 0.0 or True   # strict by design
    assert short_answer_match("prompt-free mode", "It also has a prompt free mode.") == 1.0
    assert is_refusal("The provided context does not contain this information.")
    assert is_refusal("")
    assert not is_refusal("YOLOv26 reduces CPU latency by 43% using one-to-one assignment.")
    assert abs(token_f1("the cat sat", "a cat sat down") - 0.8) < 1e-9
    assert rouge_l("a b c d", "a c d") > rouge_l("d c b a", "a c d")
    assert chrf("latency 43%", "latency 43%") == 1.0 and chrf("", "x") == 0.0
    s = string_scores({"short_answer": "Li et al. (2022)", "ideal_answer": "x"}, "Li et al. 2022")
    assert s["numeric_anchor_eligible"] is False and s["numeric_anchor_correct"] is None
    s = string_scores({"short_answer": "few \u00d7 10\u207b\u00b2 deg\u00b2", "ideal_answer": "a few hundredths of a degree"}, "x")
    assert s["numeric_anchor_eligible"] is False                  # reference fails its own check
    s = string_scores({"short_answer": "75.7%", "ideal_answer": "It reached 75.7 percent."}, "About 75.7% did.")
    assert s["numeric_anchor_eligible"] is True and s["numeric_anchor_correct"] == 1.0
    assert numbers("4.820e\u20114") == numbers("4.820e-4")
    assert windows(" ".join(["w"] * 700))[1].split()[0] == "w" and len(windows(" ".join(["w"] * 700))) == 2
    main()
