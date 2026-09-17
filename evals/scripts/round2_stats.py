"""Paired statistics and the pre-registered verdict rules for round 2.

    python evals/scripts/round2_stats.py      # self-check

Every comparison in round 2 is paired: all nine cells answer the same 400
questions, so each test compares a cell against another cell question by
question. What runs where is fixed by evals/Reports/round2_preregistration.json:

  binary metrics (span found or not)  exact two-sided sign test / McNemar
  continuous metrics                  two-sided Wilcoxon signed-rank
  effect intervals                    percentile bootstrap resampling PAPERS,
                                      not questions, because a paper can
                                      contribute more than one question
  multiplicity                        Holm within each metric's family

Verdict (rules B and D from the pre-registration): a cell is better than
another on a metric only if it wins significantly after Holm on at least 2 of
the 3 parsers, loses significantly on none, and the pooled effect across
parsers reaches the metric's threshold.
"""

import random
from math import comb, erf, sqrt


# ------------------------------------------------------------------ tests
def sign_test(a, b):
    """Exact two-sided sign test on discordant pairs (McNemar for 0/1 data).

    Returns (wins, losses, p). Pairs where the two cells scored the same carry
    no information about which is better and are dropped, which is what makes
    this exact rather than approximate.
    """
    w = sum(1 for x, y in zip(a, b) if x > y)
    l = sum(1 for x, y in zip(a, b) if x < y)
    n = w + l
    if n == 0:
        return w, l, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)
    return w, l, p


def _phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank test, normal approximation with
    continuity and tie corrections. At n ~ 400 the approximation is standard;
    it is not used below about 10 non-zero differences, where the sign test is.

    Returns (n_nonzero, W, p).
    """
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 0, 0.0, 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:                                  # average ranks within ties
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, d) if x < 0)
    W = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return n, W, 1.0
    z = (W - mean + 0.5) / sqrt(var)              # continuity correction
    return n, W, max(0.0, min(1.0, 2 * _phi(z)))


def holm(ps, alpha=0.05):
    """Holm step-down: which p-values in a family survive at alpha."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    keep = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] > alpha / (len(ps) - rank):
            break
        keep[i] = True
    return keep


def dz(a, b):
    """Paired effect size (Cohen's dz): mean difference over its own SD.

    A difference that never varies has no spread: every question moved the same
    way by the same amount, which is an unbounded effect, not a missing one.
    """
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return 0.0
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var > 0:
        return m / sqrt(var)
    return 0.0 if m == 0 else float("inf") * (1 if m > 0 else -1)


def bootstrap_diff(a, b, papers, n=10000, seed=1):
    """95% percentile interval for mean(a) - mean(b), resampling papers.

    Questions from one paper are not independent, so whole papers are drawn
    with replacement and every question of a drawn paper goes in together.
    """
    by_paper = {}
    for x, y, p in zip(a, b, papers):
        by_paper.setdefault(p, []).append(x - y)
    groups = list(by_paper.values())
    if not groups:
        return 0.0, 0.0, 0.0
    point = sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        drawn = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        total = sum(sum(g) for g in drawn)
        count = sum(len(g) for g in drawn)
        means.append(total / count if count else 0.0)
    means.sort()
    return point, means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]


# ------------------------------------------------------------------ verdicts
# The rule is "at or above the threshold". Summing a few differences in binary
# floating point can land a genuine 0.05 at 0.049999999999999996, so the
# comparison is made with a tolerance rather than letting the last bit decide.
_TOL = 1e-9


def verdict(per_parser, threshold, kind="points", unit="parsers"):
    """Rules B and D on one metric, given one entry per parser.

    per_parser: {parser: {"p": float, "holm": bool, "diff": float, "dz": float}}
    where diff is (challenger - baseline) in metric units.

    Returns "better", "worse" or "no established difference", with the reason.
    """
    sig_win = [k for k, v in per_parser.items() if v["holm"] and v["diff"] > 0]
    sig_loss = [k for k, v in per_parser.items() if v["holm"] and v["diff"] < 0]
    pooled = sum(v["diff"] for v in per_parser.values()) / len(per_parser)
    pooled_dz = sum(v["dz"] for v in per_parser.values()) / len(per_parser)
    size = abs(pooled) if kind == "points" else abs(pooled_dz)
    big = size >= threshold - _TOL
    if len(sig_win) >= 2 and not sig_loss and big and pooled > 0:
        return "better", f"significant on {len(sig_win)} of {len(per_parser)} {unit}, pooled {pooled:+.3f}"
    if len(sig_loss) >= 2 and not sig_win and big and pooled < 0:
        return "worse", f"significant loss on {len(sig_loss)} of {len(per_parser)} {unit}, pooled {pooled:+.3f}"
    why = []
    if len(sig_win) < 2 and len(sig_loss) < 2:
        why.append(f"significant on only {max(len(sig_win), len(sig_loss))} of {len(per_parser)} {unit}")
    if sig_win and sig_loss:
        why.append("wins on one parser and loses on another")
    if not big:
        why.append(f"pooled effect {pooled:+.3f} below the {threshold} threshold")
    return "no established difference", "; ".join(why)


def fixed_sequence(metrics, decide):
    """Priority order: stop at the first metric that fails to establish a
    difference. Returns the list of (metric, verdict, reason) actually tested."""
    out = []
    for m in metrics:
        v, why = decide(m)
        out.append((m, v, why))
        if v == "no established difference":
            break
    return out


if __name__ == "__main__":
    # sign test against hand-computable values
    assert sign_test([1] * 10, [0] * 10)[2] == 2 * (1 / 2 ** 10)
    assert sign_test([1, 1, 0], [1, 1, 0])[2] == 1.0          # no discordant pairs
    w, l, p = sign_test([1, 1, 1, 0], [0, 0, 0, 1])
    assert (w, l) == (3, 1) and abs(p - 0.625) < 1e-12

    # Holm: 0.01 survives, 0.03 stops the step-down at alpha/2 = 0.025
    assert holm([0.01, 0.04, 0.03], 0.05) == [True, False, False]
    assert holm([0.001, 0.002], 0.05) == [True, True]

    # Wilcoxon against scipy when it is installed
    a = [0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.8, 0.6, 0.35, 0.55, 0.15]
    b = [0.2, 0.4, 0.35, 0.6, 0.25, 0.5, 0.45, 0.5, 0.65, 0.3, 0.6, 0.25]
    n, W, p = wilcoxon(a, b)
    try:
        from scipy.stats import wilcoxon as sp
        r = sp(a, b, zero_method="wilcox", correction=True, mode="approx")
        assert abs(p - float(r.pvalue)) < 0.02, (p, float(r.pvalue))
        print(f"  wilcoxon matches scipy: p={p:.4f} vs {float(r.pvalue):.4f}")
    except ImportError:
        print(f"  scipy not installed; wilcoxon p={p:.4f} (n={n}, W={W})")

    # bootstrap: questions of one paper move together, so a lopsided paper
    # widens the interval rather than being averaged away
    pt, lo, hi = bootstrap_diff([1, 1, 1, 0], [0, 0, 0, 0], ["p1", "p1", "p1", "p2"], n=2000)
    assert abs(pt - 0.75) < 1e-9 and lo >= 0.0 and hi <= 1.0 and lo < hi

    assert dz([1, 2, 3], [0, 1, 2]) == float("inf")      # same gain every time
    assert dz([1, 2, 3], [1, 2, 3]) == 0.0
    assert abs(dz([1, 2, 3, 4], [0, 2, 2, 5]) - 0.2611) < 1e-3

    # verdict rules B and D
    sig = lambda d: {"p": 0.001, "holm": True, "diff": d, "dz": d * 5}      # noqa: E731
    ns = lambda d: {"p": 0.4, "holm": False, "diff": d, "dz": d * 5}        # noqa: E731
    # pooled = (0.08 + 0.07 + 0.02) / 3 = 0.057, over the 0.05 threshold
    v, why = verdict({"a": sig(0.08), "b": sig(0.07), "c": ns(0.02)}, 0.05)
    assert v == "better", why
    # same significance, pooled 0.047: rule D fails on the size of the effect
    v, why = verdict({"a": sig(0.06), "b": sig(0.06), "c": ns(0.02)}, 0.05)
    assert v == "no established difference" and "below the 0.05 threshold" in why
    v, why = verdict({"a": sig(0.01), "b": sig(0.01), "c": ns(0.0)}, 0.05)
    assert v == "no established difference" and "below the 0.05 threshold" in why
    v, _ = verdict({"a": sig(0.06), "b": ns(0.01), "c": ns(0.0)}, 0.05)
    assert v == "no established difference"
    v, _ = verdict({"a": sig(-0.06), "b": sig(-0.07), "c": ns(-0.02)}, 0.05)
    assert v == "worse"

    # fixed sequence stops at the first metric without a difference
    seq = fixed_sequence(["m1", "m2", "m3"],
                         lambda m: ("better", "") if m == "m1" else ("no established difference", ""))
    assert [x[0] for x in seq] == ["m1", "m2"]
    print("round2_stats self-checks passed")
