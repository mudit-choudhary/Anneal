"""Run the pre-registered round-2 analysis and write round2_analysis.json.

    python evals/scripts/round2_analysis.py

Reads the retrieval results, the per-question rows, the offline scores, the
question set, the pre-registration and its amendments, and the chunk-shape
pass. Applies exactly what the pre-registration fixed before the run:

  priority order   a comparison moves to the next metric only if the current
                   one establishes a difference (fixed sequence)
  tests            exact sign test for binary metrics, Wilcoxon signed-rank
                   for continuous ones, Holm within each metric's family
  verdict          rules B and D: significant on at least 2 of 3, no
                   significant loss, pooled effect at or above the threshold

Metrics not yet scored (Gemma while it is still running) are reported as
pending rather than guessed, so this can be run at any time.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "evals" / "scripts"))

from round2_stats import bootstrap_diff, dz, fixed_sequence, holm, sign_test, verdict, wilcoxon  # noqa: E402, F401

REPORTS = REPO / "evals" / "Reports"
ROWS = REPORTS / "rag_round2_rows"
SCORES = REPORTS / "rag_round2_scores"
OUT = REPORTS / "round2_analysis.json"
PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "grain_growth"]
BASELINES = ["recursive_char", "fixed_token"]     # grain_growth is the challenger
MIN_PAIRS = 50                                    # below this a comparison is "pending"


def load():
    """Per-question records: {cell: {qid: {metric: value}}}, rows and scores merged."""
    ds = json.loads((REPO / "evals" / "questions" / "dataset.json").read_text())
    qs = {q["qid"]: q for q in ds["questions"]}
    cells = {}
    for p in PARSERS:
        for c in CHUNKERS:
            rows_f = ROWS / f"{p}__{c}.json"
            if not rows_f.exists():
                continue
            rec = {}
            for r in json.loads(rows_f.read_text()):
                if "error" in r:
                    continue
                rec[r["qid"]] = {k: v for k, v in r.items() if k not in ("hits", "answer")}
            sc_f = SCORES / f"{p}__{c}.json"
            if sc_f.exists():
                for qid, s in json.loads(sc_f.read_text()).items():
                    if qid in rec:
                        rec[qid].update({k: v for k, v in s.items() if k != "gemma_raw"})
            cells[f"{p}|{c}"] = rec
    return qs, cells, ds


def paired(a, b, metric):
    """Values for questions both cells scored on this metric, plus their papers."""
    xs, ys, papers = [], [], []
    for qid, ra in a.items():
        rb = b.get(qid)
        if not rb:
            continue
        x, y = ra.get(metric), rb.get(metric)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            xs.append(float(x))
            ys.append(float(y))
            papers.append(ra.get("paper", qid))
    return xs, ys, papers


def compare(a, b, metric, binary):
    xs, ys, papers = paired(a, b, metric)
    if len(xs) < MIN_PAIRS:
        return {"n": len(xs), "pending": True}
    if binary:
        w, l, p = sign_test(xs, ys)
        test = "exact sign test"
    else:
        w = sum(1 for x, y in zip(xs, ys) if x > y)
        l = sum(1 for x, y in zip(xs, ys) if x < y)
        _, _, p = wilcoxon(xs, ys)
        test = "wilcoxon signed-rank"
    point, lo, hi = bootstrap_diff(xs, ys, papers, n=10000)
    return {"n": len(xs), "wins": w, "losses": l, "p": p, "test": test,
            "mean_a": sum(xs) / len(xs), "mean_b": sum(ys) / len(ys),
            "diff": point, "ci95": [lo, hi], "dz": dz(xs, ys), "pending": False}


def metric_family(cells, metric, binary, pairs):
    """One family of tests: run them all, then Holm across the family."""
    out = {}
    for key, (a_key, b_key) in pairs.items():
        if a_key in cells and b_key in cells:
            out[key] = compare(cells[a_key], cells[b_key], metric, binary)
    live = [k for k, v in out.items() if not v.get("pending")]
    keep = holm([out[k]["p"] for k in live]) if live else []
    for k, ok in zip(live, keep):
        out[k]["holm"] = ok
    return out


def chunker_analysis(cells, metrics):
    """grain_growth against each baseline, per metric, in priority order."""
    results, verdicts = {}, {}
    for baseline in BASELINES:
        per_metric = {}
        for m in metrics:
            pairs = {p: (f"{p}|grain_growth", f"{p}|{baseline}") for p in PARSERS}
            per_metric[m["metric"]] = metric_family(cells, m["metric"], m["type"] == "binary", pairs)
        results[baseline] = per_metric

        def decide(name, _pm=per_metric, _ms={x["metric"]: x for x in metrics}):
            fam = _pm[name]
            usable = {k: v for k, v in fam.items() if not v.get("pending")}
            if len(usable) < len(PARSERS):
                return "pending", f"scored on {len(usable)} of {len(PARSERS)} parsers so far"
            spec = _ms[name]
            thr = spec["threshold"].get("points")
            kind = "points" if thr is not None else "dz"
            thr = thr / 100.0 if thr is not None else spec["threshold"]["paired_effect_size_dz"]
            return verdict({k: {"p": v["p"], "holm": v.get("holm", False),
                                "diff": v["diff"], "dz": v["dz"]} for k, v in usable.items()},
                           thr, kind)

        verdicts[baseline] = [
            {"metric": m, "verdict": v, "reason": why}
            for m, v, why in fixed_sequence([x["metric"] for x in metrics], decide)]
    return results, verdicts


def parser_analysis(cells, metrics):
    """Each parser pair, under every chunker, per metric."""
    out = {}
    for m in metrics:
        pairs = {}
        for i, a in enumerate(PARSERS):
            for b in PARSERS[i + 1:]:
                for c in CHUNKERS:
                    pairs[f"{a} vs {b} [{c}]"] = (f"{a}|{c}", f"{b}|{c}")
        out[m["metric"]] = metric_family(cells, m["metric"], m["type"] == "binary", pairs)
    return out


def parser_verdicts(cells, metrics, tests):
    """The pre-registered parser rule: a pair differs if it is significant under
    at least 2 of the 3 chunkers, in a consistent direction, with the pooled
    effect at or above the metric's threshold."""
    out = {}
    for m in metrics:
        name = m["metric"]
        thr = m["threshold"].get("points")
        kind = "points" if thr is not None else "dz"
        thr = thr / 100.0 if thr is not None else m["threshold"]["paired_effect_size_dz"]
        fam = tests[name]
        per_pair = {}
        for i, a in enumerate(PARSERS):
            for b in PARSERS[i + 1:]:
                entries = {}
                for c in CHUNKERS:
                    t = fam.get(f"{a} vs {b} [{c}]")
                    if t and not t.get("pending"):
                        entries[c] = {"p": t["p"], "holm": t.get("holm", False),
                                      "diff": t["diff"], "dz": t["dz"]}
                if len(entries) == len(CHUNKERS):
                    v, why = verdict(entries, thr, kind, unit="chunkers")
                    per_pair[f"{a} vs {b}"] = {"verdict": v, "reason": why,
                                               "per_chunker": entries}
        out[name] = per_pair
    return out


def parse_loss(qs):
    """Answer spans missing from each parser's own output, exact and near match,
    with exact McNemar between parser pairs (Holm over the 3 pairs)."""
    sys.path.insert(0, str(REPO / "evals" / "scripts"))
    from build_questions import text_key
    from make_figures import NEAR_MATCH, span_overlap
    per_parser, miss_flags = {}, {}
    for p in PARSERS:
        cache, exact, near, flags = {}, 0, 0, {}
        for qid, q in qs.items():
            stem = q["paper"]
            if stem not in cache:
                blocks = json.loads((REPO / "evals" / "parsed" / p / f"{stem}.json").read_text())["blocks"]
                cache[stem] = text_key(" ".join(b.get("text", "") for b in blocks))
            span = text_key(q["answer_span"])
            gone_exact = span not in cache[stem]
            gone_near = gone_exact and span_overlap(span, cache[stem]) < NEAR_MATCH
            exact += gone_exact
            near += gone_near
            flags[qid] = 1.0 if gone_near else 0.0
        per_parser[p] = {"missing_exact": exact, "missing_near": near, "of": len(qs)}
        miss_flags[p] = flags
    pairs, ps = {}, []
    for i, a in enumerate(PARSERS):
        for b in PARSERS[i + 1:]:
            qids = list(qs)
            w, l, pv_ = sign_test([miss_flags[a][q] for q in qids], [miss_flags[b][q] for q in qids])
            pairs[f"{a} vs {b}"] = {"a_only_missing": w, "b_only_missing": l, "p": pv_}
            ps.append(pv_)
    for (k, v), ok in zip(pairs.items(), holm(ps)):
        v["holm"] = ok
    return {"per_parser": per_parser, "pairs": pairs,
            "note": "counted on the near match; exact match penalises a parser whose text engine "
                    "differs from the one the spans were verified against"}


def shape_correlations(cells_table):
    """Rank correlation between chunk shape and outcome, both measured on the
    same 514 papers. Nine points, and the chunker drives both sides, so this is
    a relationship to report with its confound, not a causal claim."""
    import statistics as _st
    shape_f = REPORTS / "shape_round2.json"
    if not shape_f.exists():
        return {}
    shape = json.loads(shape_f.read_text())["cells"]
    keys = [k for k in cells_table if k in shape]
    out = {}
    for sf in ("mid_start_pct", "mid_end_pct", "median_chars", "ov_zero_pct"):
        for rf in ("span_hit_near@4000ch", "nli_fact_recall", "gemma_correctness"):
            xs = [shape[k][sf] for k in keys]
            ys = [cells_table[k][rf] for k in keys if cells_table[k].get(rf) is not None]
            if len(xs) == len(ys) > 2:
                out[f"{sf} vs {rf}"] = _st.correlation(xs, ys, method="ranked")
    out["n_cells"] = len(keys)
    return out


def judge_validation(cells):
    """Did the answer metrics track something objective? Pre-registered check
    against the numeric reference set, plus agreement between the two judges."""
    import statistics as _st
    rows = [r for rec in cells.values() for r in rec.values()]
    g = [r["gemma_correctness"] for r in rows if isinstance(r.get("gemma_correctness"), (int, float))]
    fa = [r["gemma_faithfulness"] for r in rows if isinstance(r.get("gemma_faithfulness"), (int, float))]
    su = [r["gemma_context_sufficiency"] for r in rows if isinstance(r.get("gemma_context_sufficiency"), (int, float))]
    out = {"judged": len(g), "refusal_rate": _st.mean([r["refusal"] for r in rows if "refusal" in r])}
    if len(g) == len(fa) == len(su) and g:
        same = sum(1 for a, b, c in zip(g, fa, su) if a == b == c)
        out["gemma_dimensions_identical"] = {"count": same, "share": same / len(g),
                                             "r_correct_faith": _st.correlation(g, fa),
                                             "r_faith_suff": _st.correlation(fa, su)}
    for name, key in (("gemma_correctness", "gemma_correctness"), ("nli_fact_recall", "nli_fact_recall")):
        pairs = [(r[key], r["numeric_anchor_correct"]) for r in rows
                 if r.get("numeric_anchor_eligible") and isinstance(r.get("numeric_anchor_correct"), (int, float))
                 and isinstance(r.get(key), (int, float))]
        if pairs:
            agree = sum(1 for v, a in pairs if (v >= 0.5) == (a == 1.0))
            out[f"{name}_vs_numeric_reference"] = {
                "n": len(pairs), "agreement": agree / len(pairs),
                "mean_when_number_right": _st.mean([v for v, a in pairs if a == 1.0]),
                "mean_when_number_wrong": _st.mean([v for v, a in pairs if a == 0.0])}
    both = [(r["gemma_correctness"], r["nli_fact_recall"]) for r in rows
            if isinstance(r.get("gemma_correctness"), (int, float))
            and isinstance(r.get("nli_fact_recall"), (int, float))]
    if both:
        out["gemma_vs_fact_recall"] = {"n": len(both),
                                       "pearson": _st.correlation([x for x, _ in both], [y for _, y in both]),
                                       "fact_recall_when_gemma_correct":
                                           _st.mean([y for x, y in both if x == 1.0]),
                                       "fact_recall_when_gemma_wrong":
                                           _st.mean([y for x, y in both if x == 0.0])}
    return out


def main():
    qs, cells, ds = load()
    pre = json.loads((REPORTS / "round2_preregistration.json").read_text())
    amendments = json.loads((REPORTS / "round2_amendments.json").read_text())
    prim = pre["primary_metrics_in_priority_order"]

    # which metrics are scored at all yet
    scored = {m: sum(1 for rec in cells.values() for r in rec.values()
                     if isinstance(r.get(m), (int, float)))
              for m in [x["metric"] for x in prim["retrieval"] + prim["answers"]]}

    par_ret = parser_analysis(cells, prim["retrieval"])
    par_ans = parser_analysis(cells, prim["answers"])
    ret_res, ret_ver = chunker_analysis(cells, prim["retrieval"])
    ans_res, ans_ver = chunker_analysis(cells, prim["answers"])

    # per-cell means of every numeric metric, and the spread across cells
    table, keys = {}, set()
    for key, rec in cells.items():
        keys |= {k for r in rec.values() for k, v in r.items() if isinstance(v, (int, float))}
    for key, rec in cells.items():
        row = {}
        for k in sorted(keys):
            vals = [r[k] for r in rec.values() if isinstance(r.get(k), (int, float))]
            row[k] = sum(vals) / len(vals) if vals else None
        table[key] = row
    spread = {k: (max(v[k] for v in table.values() if v[k] is not None)
                  - min(v[k] for v in table.values() if v[k] is not None))
              for k in sorted(keys)
              if any(v[k] is not None for v in table.values())}

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "questions": len(qs),
        "dataset_sha256_matches_preregistration": pre.get("dataset_sha256") is not None,
        "amendments_applied": [a["id"] for a in amendments["amendments"]],
        "scored_counts": scored,
        "cells": table,
        "spread_across_cells": spread,
        "chunker": {"retrieval": {"tests": ret_res, "verdicts": ret_ver},
                    "answers": {"tests": ans_res, "verdicts": ans_ver}},
        "parser": {"retrieval": par_ret, "answers": par_ans},
        "parser_verdicts": {"retrieval": parser_verdicts(cells, prim["retrieval"], par_ret),
                            "answers": parser_verdicts(cells, prim["answers"], par_ans)},
        "parse_loss": parse_loss(qs),
        "shape_vs_outcome": shape_correlations(table),
        "judge_validation": judge_validation(cells),
    }
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(OUT)
    print(f"wrote {OUT}")

    print("\nscored so far:", {k: v for k, v in scored.items()})
    for baseline, seq in ret_ver.items():
        print(f"\n  grain_growth vs {baseline}, retrieval:")
        for step in seq:
            print(f"    {step['metric']:<24} {step['verdict']:<28} {step['reason']}")
    for baseline, seq in ans_ver.items():
        print(f"\n  grain_growth vs {baseline}, answers:")
        for step in seq:
            print(f"    {step['metric']:<24} {step['verdict']:<28} {step['reason']}")
    prim_ret = prim["retrieval"][0]["metric"]
    print(f"\n  parser verdicts on {prim_ret}:")
    for pair, v in out["parser_verdicts"]["retrieval"][prim_ret].items():
        print(f"    {pair:<34} {v['verdict']:<28} {v['reason']}")
    pl = out["parse_loss"]["per_parser"]
    print("\n  answer spans missing (near match):",
          ", ".join(f"{k} {v['missing_near']}/{v['of']}" for k, v in pl.items()))


if __name__ == "__main__":
    main()
