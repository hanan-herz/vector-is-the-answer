"""Paired significance off persisted per-row preds (no model, CPU-only).

Arm A runs save ``cache/*_rowpreds.npz`` next to the run (see bench.py,
``mlp_pred_nets`` + the ``_rowpreds`` persistence block): ``y_true``,
``readout.mlp`` (seed-ensemble majority vote) and every ``loop.*`` arm, all
aligned to the same val rows. This script turns that artifact into the
paired-test table the paper needs:

  * per (readout, loop-arm) pair on the SAME rows:
      - accuracy of each side
      - McNemar exact binomial on the discordant pairs
        (readout right/loop wrong vs loop right/readout wrong)
      - paired bootstrap 95% CI on acc(readout) - acc(loop)
  * consecutive-arm comparison (loop.k vs loop.(k/2)) to test the plateau
    claim from paper/budget-matched-loop.md

USAGE
  python rowpreds_stats.py \
      cloud_bench_cache/Qwen__Qwen3-0.6B/boolq/cache/vec_t0_v3270_l27_p8192_rowpreds.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: b = A-right/B-wrong, c = B-right/A-wrong."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact: 2 * P(X <= k), X ~ Bin(n, 0.5), capped at 1
    return float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))


def boot_ci(a_correct: np.ndarray, b_correct: np.ndarray, n_boot: int = 10000,
            seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap on the accuracy difference a - b. Rows resampled together."""
    rng = np.random.default_rng(seed)
    diff = a_correct.astype(float) - b_correct.astype(float)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    ds = diff[idx].mean(axis=1)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def pair_report(name_a: str, pa: np.ndarray, name_b: str, pb: np.ndarray,
                y: np.ndarray) -> dict:
    ca, cb = (pa == y), (pb == y)
    both = int(np.sum(ca & cb))
    a_only = int(np.sum(ca & ~cb))   # readout right, loop wrong
    b_only = int(np.sum(~ca & cb))   # loop right, readout wrong
    neither = int(np.sum(~ca & ~cb))
    d, lo, hi = boot_ci(ca, cb)
    return {
        "pair": f"{name_a} vs {name_b}",
        "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
        "diff": d, "ci95": [lo, hi],
        "both_right": both, "a_only": a_only, "b_only": b_only,
        "neither": neither,
        "mcnemar_p": mcnemar_exact(a_only, b_only),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to *_rowpreds.npz")
    ap.add_argument("--readout", default="readout.mlp")
    ap.add_argument("--pairs", default=None,
                    help="explicit comma list of A:B key pairs to compare "
                         "(e.g. readout.mlp.L18:readout.mlp.L27 for a "
                         "layer-vs-layer McNemar off a sweep rowpreds npz). "
                         "Overrides the default readout-vs-loops mode.")
    ap.add_argument("--json", default=None, help="also write results to this path")
    args = ap.parse_args()

    d = np.load(args.npz)
    y = d["y_true"]

    if args.pairs:
        reports = []
        print(f"rows: {len(y)}   keys: {sorted(d.keys())}\n")
        for pair in args.pairs.split(","):
            ka, kb = pair.split(":")
            r = pair_report(ka, d[ka], kb, d[kb], y)
            reports.append(r)
            sig = "***" if r["mcnemar_p"] < 0.001 else "**" if r["mcnemar_p"] < 0.01 else "*" if r["mcnemar_p"] < 0.05 else "ns"
            print(f"{ka} ({r['acc_a']:.4f}) vs {kb} ({r['acc_b']:.4f}): "
                  f"diff {r['diff']:+.4f}  CI95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
                  f"discordant a-only={r['a_only']} b-only={r['b_only']}  "
                  f"McNemar p={r['mcnemar_p']:.2e} {sig}")
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({"npz": args.npz, "n": len(y),
                           "reports": reports}, fh, indent=2)
            print(f"\nwrote {args.json}")
        return
    loops = sorted([k for k in d.keys() if k.startswith("loop.")],
                   key=lambda k: (k != "loop.zero", int(k.split(".")[1]) if k.split(".")[1].isdigit() else -1))
    print(f"rows: {len(y)}   arms: {loops}   readout: {args.readout}\n")

    reports = []
    pr = d[args.readout]
    for lk in loops:
        r = pair_report(args.readout, pr, lk, d[lk], y)
        reports.append(r)
        sig = "***" if r["mcnemar_p"] < 0.001 else "**" if r["mcnemar_p"] < 0.01 else "*" if r["mcnemar_p"] < 0.05 else "ns"
        print(f"{args.readout} ({r['acc_a']:.4f}) vs {lk} ({r['acc_b']:.4f}): "
              f"diff {r['diff']:+.4f}  CI95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"discordant a-only={r['a_only']} b-only={r['b_only']}  "
              f"McNemar p={r['mcnemar_p']:.2e} {sig}")

    # plateau test: consecutive loop arms
    num = [k for k in loops if k != "loop.zero"]
    if len(num) >= 2:
        print()
        for prev, cur in zip(num, num[1:]):
            r = pair_report(cur, d[cur], prev, d[prev], y)
            reports.append(r)
            sig = "***" if r["mcnemar_p"] < 0.001 else "**" if r["mcnemar_p"] < 0.01 else "*" if r["mcnemar_p"] < 0.05 else "ns"
            print(f"{cur} ({r['acc_a']:.4f}) vs {prev} ({r['acc_b']:.4f}): "
                  f"diff {r['diff']:+.4f}  CI95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
                  f"McNemar p={r['mcnemar_p']:.2e} {sig}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"npz": args.npz, "n": len(y), "reports": reports}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
