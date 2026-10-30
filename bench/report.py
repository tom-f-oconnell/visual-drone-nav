"""Tables from match_bench.py output: rates at the confidence threshold that keeps wrong fixes rare.

Usage: report.py OUT_DIR [--max-wrong 0.01] [--gross-m 20] [--matchers SPEC ...]

A fix is correct within 10 m of the truth and wrong otherwise. For each
matcher and search radius, the threshold is the lowest score at which wrong
fixes are at most --max-wrong of all frames, pooled over every site. The
matcher's single threshold is the higher of its per-radius ones, since the
navigator uses one value whatever the search radius. Because thresholds are
fitted to the same frames they are scored on, the leave-one-site-out (LOSO)
columns score each site with the threshold fitted to the other sites instead.
A second table counts only gross errors (--gross-m or more) as wrong, with
fixes between 10 m and that as near misses, since the navigator's filter
copes with a 10 to 20 m fix far better than with a fix on the wrong field.
Writes OUT_DIR/report.md and prints it.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from sites import BY_NAME, SITES

CORRECT_M = 10.0


def load(out: Path) -> dict[str, list[dict]]:
    by_matcher = defaultdict(list)
    for p in sorted(out.glob("*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                r["radius_m"] = float(r["radius_m"])
                r["fixed"] = r["fixed"] == "1"
                r["err_m"] = float(r["err_m"]) if r["fixed"] else np.inf
                r["score"] = float(r["score"]) if r["fixed"] else -np.inf
                r["ms"] = float(r["ms"]) if r["ms"] else np.nan
                by_matcher[r["matcher"]].append(r)
    return by_matcher


def rates(rows: list[dict], thr: float, wrong_m: float = CORRECT_M) -> dict:
    """Fractions of frames with correct, near (10 m to wrong_m), wrong and no fix at threshold thr."""
    n = len(rows)
    err = np.array([r["err_m"] for r in rows])
    score = np.array([r["score"] for r in rows])
    used = np.array([r["fixed"] for r in rows], bool) & (score >= thr)
    ok = used & (err < CORRECT_M)
    bad = used & (err >= wrong_m)
    e = err[ok]
    return {
        "n": n, "correct": ok.sum() / n, "near": (used & ~ok & ~bad).sum() / n, "wrong": bad.sum() / n,
        "none": 1 - used.sum() / n, "n_ok": int(ok.sum()), "n_wrong": int(bad.sum()),
        "err_med": float(np.median(e)) if len(e) else np.nan,
        "err_p90": float(np.percentile(e, 90)) if len(e) else np.nan,
    }


def threshold(rows: list[dict], max_wrong: float, wrong_m: float = CORRECT_M) -> float:
    """Lowest score threshold with fixes wrong by wrong_m or more <= max_wrong of all frames."""
    n = len(rows)
    fixed = sorted((r["score"], r["err_m"] >= wrong_m) for r in rows if r["fixed"])
    wrong_above = sum(w for _, w in fixed)
    for i, (s, w) in enumerate(fixed):
        if (i == 0 or s != fixed[i - 1][0]) and wrong_above <= max_wrong * n:
            return s
        wrong_above -= w
    return np.inf


def loso(rows: list[dict], max_wrong: float) -> tuple[float, float]:
    """Correct and wrong fractions when each site is scored with the threshold fitted to the others."""
    ok = bad = 0
    for site in {r["site"] for r in rows}:
        thr = threshold([r for r in rows if r["site"] != site], max_wrong)
        a = rates([r for r in rows if r["site"] == site], thr)
        ok, bad = ok + a["n_ok"], bad + a["n_wrong"]
    return ok / len(rows), bad / len(rows)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def fmt(thr: float) -> str:
    return "never" if np.isinf(thr) else f"{thr:.3g}"


def report(by_matcher: dict[str, list[dict]], max_wrong: float, gross_m: float,
           matchers: list[str] | None = None) -> str:
    matchers = matchers or sorted(by_matcher)
    radii = sorted({r["radius_m"] for rows in by_matcher.values() for r in rows})
    sites = [s.name for s in SITES if any(r["site"] == s.name for rows in by_matcher.values() for r in rows)]
    lines = [f"{len(sites)} sites. Correct: within {CORRECT_M:.0f} m. Threshold: lowest score with wrong fixes "
             f"<= {pct(max_wrong)} of frames, pooled over sites.", ""]

    thr = {}
    lines += ["## Overall", "",
              "| matcher | radius | frames | raw correct / wrong | threshold | correct | wrong | none | "
              "LOSO correct / wrong | err median / p90 (m) | ms median |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in matchers:
        rows = by_matcher[m]
        per_r = {rad: threshold([r for r in rows if r["radius_m"] == rad], max_wrong) for rad in radii}
        thr[m] = max(per_r.values())
        for rad in radii:
            rr = [r for r in rows if r["radius_m"] == rad]
            raw, a = rates(rr, -np.inf), rates(rr, per_r[rad])
            lo = loso(rr, max_wrong)
            ms = np.nanmedian([r["ms"] for r in rr])
            lines.append(f"| {m} | {rad:.0f} m | {a['n']} | {pct(raw['correct'])} / {pct(raw['wrong'])} | "
                         f"{fmt(per_r[rad])} | {pct(a['correct'])} | {pct(a['wrong'])} | {pct(a['none'])} | "
                         f"{pct(lo[0])} / {pct(lo[1])} | {a['err_med']:.1f} / {a['err_p90']:.1f} | {ms:.0f} |")

    lines += ["", f"## Overall, counting only errors of {gross_m:.0f} m or more as wrong", "",
              f"| matcher | radius | threshold | correct | near miss (10-{gross_m:.0f} m) | wrong | none |",
              "|---|---|---|---|---|---|---|"]
    for m in matchers:
        for rad in radii:
            rr = [r for r in by_matcher[m] if r["radius_m"] == rad]
            t = threshold(rr, max_wrong, gross_m)
            a = rates(rr, t, gross_m)
            lines.append(f"| {m} | {rad:.0f} m | {fmt(t)} | {pct(a['correct'])} | {pct(a['near'])} | "
                         f"{pct(a['wrong'])} | {pct(a['none'])} |")

    lines += ["", "## Per site: correct (wrong count) at the matcher's single threshold; "
              "raw correct after the slash", "",
              "Single thresholds (the higher of the per-radius ones): "
              + ", ".join(f"{m} {fmt(thr[m])}" for m in matchers), ""]
    head = "| site | terrain | " + " | ".join(f"{m} {rad:.0f} m" for m in matchers for rad in radii) + " |"
    lines += [head, "|" + "---|" * (2 + len(matchers) * len(radii))]
    for s in sites:
        cells = []
        for m in matchers:
            for rad in radii:
                rr = [r for r in by_matcher[m] if r["site"] == s and r["radius_m"] == rad]
                if rr:
                    a, raw = rates(rr, thr[m]), rates(rr, -np.inf)
                    cells.append(f"{pct(a['correct'])} ({a['n_wrong']}) / {pct(raw['correct'])}")
                else:
                    cells.append("-")
        lines.append(f"| {s} | {BY_NAME[s].terrain} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-wrong", type=float, default=0.01)
    ap.add_argument("--gross-m", type=float, default=20.0)
    ap.add_argument("--matchers", nargs="*")
    args = ap.parse_args(argv)
    text = report(load(args.out), args.max_wrong, args.gross_m, args.matchers)
    (args.out / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
