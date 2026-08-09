"""Paired significance: one-pass head vs fair loop on matched rows (Thread 1).

PURPOSE
The headline "loop never cleanly beats the readout" rests on ties / ~1pt gaps
with no paired test and no error bars. This puts confidence intervals + McNemar
on the comparison, head vs loop, row-aligned, per task and pooled.

METHOD (per task, on the SAME val rows)
  - one-pass head: saved final-residual MLP (full-train / shelf head) applied
    to the cached last-token val residuals -> per-row pred
  - loop: fair next-token continuation over the closed answer set
    (loop.k8 / loop.zero), re-run per row, aligned to val -> per-row pred
  - then on matched (head_pred, loop_pred, gold) triples:
      * McNemar exact binomial on (head right, loop wrong) vs (loop right,
        head wrong) -> is either direction significant?
      * paired bootstrap CI on the accuracy difference head - loop

Also pools all three tasks (each row is an independent within-task pair) to lift
power while staying a valid paired test.

USAGE (local Qwen3-0.6B; residuals + heads already on disk)
  python paired_test.py --size 0.6B --max-val 200
Set --arc-full to also score the full ARC test cache (1165 rows) for a larger
single-cell n.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats
import torch

from common import load_model, n_layers as discover_layers, resolve_model
from bench import (
    CACHE_DIR,
    LOOP_PAD_MAX,
    RESULTS_DIR,
    encode_labels,
    fmt_example,
    load_cached_vectors,
    loop_report,
    model_slug,
    task_dirs,
)
from multihead_route import (
    TASKS,
    ensure_residuals,
    find_head,
    head_forward,
    load_head_npz,
)
from tasks import get_task

# Full official sizes, for --full-val at 0.6B.
FULL_VAL = {"boolq": 3270, "ruletaker": 2000, "arc": 1165}
# Train sizes used if --full-val (residuals must be extracted for BoolQ/RuleTaker)
_FULL_TRAIN_FOR_VAL = {"boolq": 9427, "ruletaker": 2000, "arc": 1117}

BINARY_KEYS = {"boolq": ("loop.zero",), "ruletaker": ("loop.zero",)}
# For binary tasks the fair few-shot baseline is loop.k8; ARC also loop.k8.
FAIR_KEYS = {"boolq": "loop.8", "ruletaker": "loop.8", "arc": "loop.8"}
ZERO_KEY = "loop.zero"


def load_train_rows(task, max_train):
    spec = get_task(task)
    rng = np.random.default_rng(0)
    train, _, _, _, _ = spec.load(max_train, None, rng)
    return train


def load_val_rows(task):
    """Reload the same val rows the residuals were extracted on (fixed seeds)."""
    spec = get_task(task)
    rng = np.random.default_rng(0)
    _, val, _, _, _ = spec.load(None, None, rng)
    return val


def head_pred_from_task(task, size, layer, max_val, model=None, tok=None,
                        device="mps", batch=8, max_train=None):
    """Per-row one-pass MLP pred: ensure residuals, apply saved head.

    If the cached residual for (task, max_val) is absent, extracts it with a
    forward pass (needs model/tok). max_train used for extraction if given.
    """
    dirs = task_dirs(CACHE_DIR, size, task)
    cached = load_cached_vectors(
        max_train, max_val, ["last_va", "yva"], cache_dir=dirs["cache"],
        layer=layer)
    if cached is not None:
        X = np.asarray(cached["last_va"][:max_val], dtype=np.float32)
        y = np.asarray(cached["yva"][:max_val])
    elif model is not None:
        res = ensure_residuals(
            size, task, max_train, max_val, model, tok, layer, batch, device)
        X = np.asarray(res["last_va"], dtype=np.float32)
        y = np.asarray(res["yva"])
    else:
        raise SystemExit(
            f"no cached residuals for {task}@v{max_val} and no model to extract")
    pred, conf, _ = head_forward(load_head_npz(find_head(size, task, layer)), X)
    return pred.astype(np.int64), y.astype(np.int64)


def paired_metrics_from_flags(h_c, l_c):
    """Given row-aligned correctness booleans (head, loop), return stats."""
    h_c = np.asarray(h_c, dtype=bool)
    l_c = np.asarray(l_c, dtype=bool)
    b = int(np.sum(h_c & ~l_c))   # head right, loop wrong
    c = int(np.sum(l_c & ~h_c))   # loop right, head wrong
    n = len(h_c)
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        k0 = min(b, c)
        # Two-sided exact McNemar: P(X <= k0) under H0:D~Bin(nd, .5), ×2.
        p = 2.0 * stats.binom.cdf(k0, n_disc, 0.5)
        p = min(p, 1.0)
    diff = float(h_c.mean() - l_c.mean())
    rng = np.random.default_rng(0)
    D = h_c.astype(np.float64) - l_c.astype(np.float64)
    boot = np.array([
        rng.choice(D, size=n, replace=True).mean() for _ in range(10_000)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": int(n),
        "head_acc": float(h_c.mean()),
        "loop_acc": float(l_c.mean()),
        "diff_head_minus_loop": diff,
        "diff_ci95": [float(lo), float(hi)],
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_discordant": n_disc,
        "mcnemar_p": float(p),
        "head_better": b > c,
        "loop_better": c > b,
        "significant95": bool((lo > 0) or (hi < 0)),
    }


def main():
    ap = argparse.ArgumentParser(description="Paired significance head vs loop")
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--full-val", action="store_true",
                    help="full per-task val (BoolQ 3270 / RuleTaker 2000 / "
                         "ARC 1165); extracts residuals for BoolQ/RuleTaker")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for L in (27, 35, 42):
        if any(find_head(args.size, t, L) for t in TASKS):
            layer = L
            break
    else:
        model, tok = load_model(args.size, device="mps")
        layer = discover_layers(model) - 1
        del model

    # Resolve per-task sizes
    sizes = {}
    if args.full_val:
        sizes = dict(FULL_VAL)
    else:
        sizes = {t: args.max_val for t in TASKS}
    print(f"model={args.size} layer={layer} sizes={sizes} full_val={args.full_val}")

    # Need model for loop forward (always) and possibly residual extraction.
    model, tok = load_model(args.size, device="mps")
    # Pre-extract residuals for tasks missing the full cache so head_pred works.
    # (cheap to do once; avoids interleaving with the loop forward)
    for t, nv in sizes.items():
        if args.full_val:
            dirs = task_dirs(CACHE_DIR, args.size, t)
            if load_cached_vectors(
                    _FULL_TRAIN_FOR_VAL[t], nv, ["last_va", "yva"],
                    cache_dir=dirs["cache"], layer=layer) is None:
                print(f"  [extract] {t} full residuals t{_FULL_TRAIN_FOR_VAL[t]}/v{nv}...")
                ensure_residuals(
                    args.size, t, _FULL_TRAIN_FOR_VAL[t], nv,
                    model, tok, layer, args.batch, "mps")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reload model for the loop forward.
    model, tok = load_model(args.size, device="mps")

    rows_out = {}
    pooled_flags = {"loop.8": {"h": [], "l": [], "y": []},
                    "loop.zero": {"h": [], "l": [], "y": []}}

    for task, nval in sizes.items():
        print(f"\n=== {task} (val {nval}) ===")
        spec = get_task(task)
        answer_set = tuple(spec.answer_set)
        mtrain = _FULL_TRAIN_FOR_VAL[task] if args.full_val else 200

        # one-pass head
        head_pred, y = head_pred_from_task(
            task, args.size, layer, nval, max_train=mtrain)
        head_pred = head_pred[:nval]
        y = y[:nval]

        # fit / exemplar train rows for few-shot
        train = load_train_rows(task, mtrain)
        _, val, _, _, _ = spec.load(None, None, np.random.default_rng(0))
        val = val[:nval]
        texts = [fmt_example(r) for r in val]

        res, preds_by_key = loop_report(
            model, tok, train, val, k_shots=(0, 8), batch=args.batch,
            pad_max=LOOP_PAD_MAX, answer_set=answer_set)

        print("  loop overall:", {k: round(v, 4) for k, v in res.items()})

        for key, preds in preds_by_key.items():
            arr = np.asarray(preds, dtype=np.int64)[:nval]
            m = paired_metrics_from_flags(head_pred == y, arr == y)
            rows_out[f"{task}:{key}"] = m
            # accumulate raw flags for a true pooled by-row test
            pf = pooled_flags[key]
            pf["h"].extend((head_pred == y).tolist())
            pf["l"].extend((arr == y).tolist())
            pf["y"].extend(y.tolist())
            sig = "*" if m["significant95"] else ""
            print(f"  head vs {key:8s} head={m['head_acc']:.3f} "
                  f"loop={m['loop_acc']:.3f} "
                  f"Δ={m['diff_head_minus_loop']:+.3f} "
                  f"CI={m['diff_ci95'][0]:+.3f}..{m['diff_ci95'][1]:+.3f} "
                  f"McNemar p={m['mcnemar_p']:.3f} {sig}")

    # TRUE pooled by-row McNemar + bootstrap over all three tasks
    print("\n=== POOLED (BoolQ + RuleTaker + ARC, matched by-row) ===")
    for key in ("loop.8", "loop.zero"):
        pf = pooled_flags[key]
        H = np.array(pf["h"], dtype=bool)
        L = np.array(pf["l"], dtype=bool)
        m = paired_metrics_from_flags(H, L)
        sig = "*" if m["significant95"] else ""
        print(f"  head vs {key:8s} head={m['head_acc']:.3f} "
              f"loop={m['loop_acc']:.3f} "
              f"Δ={m['diff_head_minus_loop']:+.3f} "
              f"CI={m['diff_ci95'][0]:+.3f}..{m['diff_ci95'][1]:+.3f} "
              f"McNemar p={m['mcnemar_p']:.3f} {sig}")
        rows_out[f"POOLED:{key}"] = m

    # Save
    out = args.out or os.path.join(
        RESULTS_DIR,
        f"paired_test_{model_slug(args.size)}"
        f"{'_fullval' if args.full_val else f'_v{args.max_val}'}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"meta": {"model": args.size, "model_id": resolve_model(args.size),
                            "layer": layer,
                            "sizes": sizes,
                            "full_val": args.full_val},
                   "pairs": rows_out, "config": list(sizes.items())},
                  fh, indent=2, default=str)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()