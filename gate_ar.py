"""Binary gate pilot: head-read vs allow-autoregression (AR).

Question: from the final residual h, can we decide CONFIDENTLY that a prompt is
covered by a closed-set head (serve it) vs must autoregress?

Two gates:
  geometric   (no open-labeled training — the honest deployment case)
      coveredness = max over closed tasks of Mahalanobis proximity to that
      task's TRAIN residual manifold. Closed prompts should sit near, open
      prompts far. Threshold → keep (head) vs escalate (AR).
  trained     (bound if we had a small open-labeled set)
      logistic on h: covered vs open, fit on closed-train + open-train.

Eval on held-out: closed val (should keep) ∪ open eval (should escalate).
Reports:
  keep@T   closed kept as heads (want ~1)
  keep@T   open kept as heads (false-ship; want ~0)
  correct-answer when kept (tie to the pilot answer numbers)
  escalate rate / false-escalate (closed wrongly sent to AR)

Usage:
  python gate_ar.py --size 0.6B --max-train 400 --max-val 200
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from common import load_model, n_layers as discover_layers, resolve_model
from common import (
    CACHE_DIR,
    RESULTS_DIR,
    load_cached_vectors,
    model_slug,
    task_dirs,
)
from bench import (
    to_vecs,
)
from multihead_route import (
    TASKS,
    find_head,
    fit_mahalanobis,
    head_forward,
    load_head_npz,
    mahalanobis_score,
)

# Open-ended, AR-required prompts. Deliberately diverse: essay, code, dialogue,
# explanation, summary, open QA, creative, advice. Includes some that LOOK
# closed-form ("Is X good?") but have no Yes/No ground truth — the hard case.
OPEN_PROMPTS = [
    # explanation / expository
    "Explain how the digestive system works from mouth to intestines.",
    "Why does the sky change color at sunset and sunrise?",
    "Describe how a refrigerator keeps food cold using a heat pump cycle.",
    "What are the causes and effects of the Industrial Revolution?",
    "How does a bill become a law in the United States?",
    "Explain the difference between open-source and proprietary software.",
    "What is the relationship between inflation and unemployment?",
    "Why do we have seasons, and how does axial tilt cause them?",
    "Describe how neural networks are trained using backpropagation.",
    "What causes ocean tides and why are there two high tides a day?",
    # essay / opinion / advice
    "Write an essay arguing whether remote work is better than office work.",
    "Should governments invest more in public transportation? Give reasons.",
    "What career advice would you give to someone entering tech in 2026?",
    "Write a thoughtful response to the prompt 'is money the root of all evil?'",
    "Argue for or against universal basic income with evidence.",
    "Draft tips for maintaining work-life balance as a software engineer.",
    "Do you think artificial intelligence will eliminate most jobs? Explain.",
    "What makes a good leader, and how do you develop leadership skills?",
    "Is it ethical to use AI to grade student essays? Discuss.",
    "Offer practical advice for learning a new language as an adult.",
    # creative / narrative
    "Write a short story about a robot that learns to paint.",
    "Compose a haiku about the ocean at midnight.",
    "Invent a product that solves commuting pain and describe it.",
    "Write a scene where two old friends meet after a decade apart.",
    "Create a fictional planet and describe its ecosystem and culture.",
    "Write a poem about change and growing older.",
    "Describe a peaceful morning in a small coastal town.",
    "Write dialogue between a skeptical customer and a friendly barista.",
    "Come up with three creative names for a coffee shop and explain them.",
    "Write a letter apologizing to a friend with warmth and honesty.",
    # code / technical task
    "Write a Python function to detect if a string is a palindrome.",
    "Explain how to debounce a button press on an Arduino.",
    "Write SQL to find duplicate emails in a users table.",
    "How would you design a URL shortener? Walk through the parts.",
    "Give me a bash one-liner to find the largest files in a directory.",
    "Write a regex to validate an email address and explain it.",
    "Describe how to set up continuous integration for a small team.",
    "Explain the tradeoffs of microservices versus a monolith.",
    "Write pseudocode for an in-memory LRU cache.",
    "How do you debug a memory leak in a long-running service?",
    # summarize / transform
    "Summarize the plot of a typical heist movie in two sentences.",
    "Rewrite this sentence more concisely: 'due to the fact that it was raining'.",
    "Turn this bullet list into a short paragraph for a resume.",
    "Condense the main idea of the statement of work into one line.",
    "Paraphrase the phrase 'think outside the box' in clearer words.",
    "Give a one-paragraph summary of what photosynthesis does.",
    "Translate the idea of 'serendipity' into a simple analogy.",
    "Reduce this long email to a single-sentence update.",
    # open QA / chat
    "What are you? How do you think and make decisions?",
    "Tell me more about yourself and your capabilities.",
    "What should I cook for dinner tonight, and why?",
    "Can you recommend a good book to read this summer?",
    "What is the meaning of life?",
    "How do I know if a startup idea is worth pursuing?",
    "What are some healthy habits that are easy to stick with?",
    "How should I prepare for a job interview at a tech company?",
    "What music would you suggest for focusing while coding?",
    "Where should I travel for a three-day weekend on a budget?",
    # closed-LOOKING but open / no single answer
    "Check my resume for weaknesses and tell me how to improve it.",
    "Is this idea (no label given) good?",  # empty premise
    "Should I accept the job offer or stay where I am?",
    "Is a 15% market share increase good or bad for the company?",
    "Review the following paragraph for clarity and tone.",
    "What's the right way to apologize in this situation?",
    "Are these two algorithms equivalent, and which is faster?",
    "Should we postpone the launch to fix the remaining bugs?",
    "Is it better to learn multiple languages or master one?",
    "Can you proofread and improve the grammar of this draft?",
]

# Split open prompts into fit (train the trained gate) and eval (never seen).
OPEN_FIT, OPEN_EVAL = OPEN_PROMPTS[:40], OPEN_PROMPTS[40:]


def extract(model, tok, texts, layer, batch=8, device="mps", label=""):
    Xlast, _ = to_vecs(model, tok, texts, layer, batch=batch, label=label)
    return Xlast.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Head-vs-AR binary gate pilot")
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--max-train", type=int, default=400)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"model={args.size} device={device} "
          f"closed train={args.max_train}/task val={args.max_val}/task")

    # layer from existing heads else discover
    layer = None
    for t in TASKS:
        for L in (27, 35, 42):
            if find_head(args.size, t, L):
                layer = L
                break
        if layer is not None:
            break

    closed_tr, closed_va = [], []
    for t in TASKS:
        dirs = task_dirs(CACHE_DIR, args.size, t)
        cached = load_cached_vectors(
            args.max_train, args.max_val,
            ["last_tr", "last_va", "ytr", "yva"],
            cache_dir=dirs["cache"], layer=layer if layer is not None else 27,
            batch=args.batch)
        if cached is None:
            raise SystemExit(f"missing residual cache for {t} — run multihead_route.py first")
        closed_tr.append(cached["last_tr"].astype(np.float32))
        closed_va.append(cached["last_va"].astype(np.float32))
        if layer is None:
            layer = 27

    Xclosed_tr = np.concatenate(closed_tr, axis=0)   # 3*max_train
    Xclosed_va = np.concatenate(closed_va, axis=0)   # 3*max_val
    n_tr_task = len(closed_tr[0]) if closed_tr else 0
    n_va_task = len(closed_va[0]) if closed_va else 0
    print(f"closed train {Xclosed_tr.shape[0]} (={n_tr_task}/task), "
          f"val {Xclosed_va.shape[0]} (={n_va_task}/task)")

    # Extract open residuals (forward pass)
    model, tok = load_model(args.size, device=device)
    if layer is None:
        layer = discover_layers(model) - 1
    print(f"target layer {layer}; extracting {len(OPEN_PROMPTS)} open prompts...")
    Xopen = extract(model, tok, OPEN_PROMPTS, layer, batch=args.batch,
                    device=device, label="open")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    Xopen_fit = Xopen[:len(OPEN_FIT)]
    Xopen_eval = Xopen[len(OPEN_FIT):]
    print(f"open fit {len(Xopen_fit)} eval {len(Xopen_eval)}")
    np.save(os.path.join(RESULTS_DIR, f"open_prompts_{model_slug(args.size)}_l{layer}.npy"),
            Xopen)

    # ---------------------------------------------------------------- heads --
    heads = {}
    n_classes_of = {}
    for t in TASKS:
        p = find_head(args.size, t, layer)
        heads[t] = load_head_npz(p)
        n_classes_of[t] = heads[t]["n_classes"]

    # ----------------------------- geometric gate (no open training) --------
    # coveredness = max over tasks of mahalanobis proximity to that task train
    maha = {t: fit_mahalanobis(closed_tr[i]) for i, t in enumerate(TASKS)}
    def coveredness(X):
        scores = np.stack([mahalanobis_score(maha[t], X) for t in TASKS], axis=1)
        return scores.max(axis=1)   # high = near some covered manifold

    cov_closed_tr = coveredness(Xclosed_tr)
    cov_closed = coveredness(Xclosed_va)
    cov_open = coveredness(Xopen_eval)
    # Threshold anchored to the COVERED-TRAIN distribution (the honest,
    # no-open-label case): keep iff at-or-above a percentile of train coveredness.
    geo_thresh = {
        q: float(np.percentile(cov_closed_tr, q)) for q in (50, 80, 95)
    }

    # ----------------------------- trained gate (has open label) ------------
    yfit = np.concatenate([
        np.zeros(len(Xclosed_tr), dtype=int),      # 0 = covered (head)
        np.ones(len(Xopen_fit), dtype=int),        # 1 = open (AR)
    ])
    Xfit = np.vstack([Xclosed_tr, Xopen_fit])
    sc = StandardScaler().fit(Xfit)
    gate = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                              class_weight="balanced")
    gate.fit(sc.transform(Xfit), yfit)
    p_closed_tr = gate.predict_proba(sc.transform(Xclosed_tr))[:, 1]
    p_open_tr = gate.predict_proba(sc.transform(Xopen_fit))[:, 1]
    p_closed = gate.predict_proba(sc.transform(Xclosed_va))[:, 1]   # AR prob
    p_open = gate.predict_proba(sc.transform(Xopen_eval))[:, 1]

    # ------------------------------- metrics --------------------------------
    def report(tag, s_closed, s_open, thresholds=(0.5, 0.7, 0.9)):
        print(f"\n=== gate: {tag} (higher score = closed/keep) ===")
        print("   thresh | closed-keep | open-escalate | false-ship | false-escalate")
        for T in thresholds:
            keep_c = float((s_closed >= T).mean())
            keep_o = float((s_open >= T).mean())
            escalate_o = 1.0 - keep_o
            false_ship = keep_o                      # open wrongly sent to head
            false_esc = 1.0 - keep_c                # closed wrongly sent to AR
            print(f"   {T:.2f}   | {keep_c:.3f}     | {escalate_o:.3f}       | "
                  f"{false_ship:.3f}    | {false_esc:.3f}")
        return

    def report_scaled(tag, s_closed, s_open, p50, p95):
        print(f"\n=== gate: {tag} ===")
        print("   rule                             | closed-keep | "
              "open-escalate | false-ship | false-escalate")
        for name, T in [("≥ closed-train p50", p50), ("≥ closed-train p95", p95)]:
            keep_c = float((s_closed >= T).mean())
            keep_o = float((s_open >= T).mean())
            print(f"   {name:30s} | {keep_c:.3f}     | "
                  f"{1.0-keep_o:.3f}         | {keep_o:.3f}    | {1.0-keep_c:.3f}")
        # min+range-normalized operating points
        lo = min(float(s_closed.min()), float(s_open.min()))
        hi = max(float(s_closed.max()), float(s_open.max()))
        norm = lambda x: (x - lo) / (hi - lo) if hi > lo else x * 0
        print("   (z-normalized operating pts)     |         | "
              "           |          |")
        for zT in (0.5, 0.75, 0.9):
            T = lo + zT * (hi - lo)
            keep_c = float((s_closed >= T).mean())
            keep_o = float((s_open >= T).mean())
            print(f"   norm@{zT:.2f} (={T:9.2f})         | {keep_c:.3f}     | "
                  f"{1.0-keep_o:.3f}         | {keep_o:.3f}    | {1.0-keep_c:.3f}")
        return

    # geometric (threshold anchored to covered-train percentiles)
    print("\n=== geometric: thresholds anchored to covered-TRAIN coveredness ===")
    for q, T in geo_thresh.items():
        keep_c = float((cov_closed >= T).mean())
        spread = float((cov_open >= T).mean())
        print(f"  p{q:>3} train cov={T:9.2f} | closed-keep={keep_c:.3f} "
              f"open-false-ship={spread:.3f}")
    report_scaled("geometric (Mahalanobis, no open label)",
                  cov_closed, cov_open, geo_thresh[50], geo_thresh[95])
    # trained (score to keep = 1 - p(open))
    report("trained logistic (AR prob → 1-conf)",
           1.0 - p_closed, 1.0 - p_open,
           thresholds=(0.5, 0.7, 0.9))

    # --------------------------- answer correctness when kept --------------
    # geometric keep anchored to closed-TRAIN p50 (no open label on train)
    Tgeo = geo_thresh[50]
    keep_g = cov_closed >= Tgeo
    print(f"\n=== closed val: answer accuracy when routed to head (geometric "
          f"≥ closed-train p50 = {Tgeo:.0f}) ===")
    ans_by_task = {}
    for j, t in enumerate(TASKS):
        dirs = task_dirs(CACHE_DIR, args.size, t)
        cached = load_cached_vectors(
            args.max_train, args.max_val, ["yva"],
            cache_dir=dirs["cache"], layer=layer, batch=args.batch)
        yva = np.asarray(cached["yva"])
        sl = np.arange(j * n_va_task, (j + 1) * n_va_task)
        pred, conf, _ = head_forward(heads[t], Xclosed_va[sl])
        acc_all = float((pred == yva).mean())
        k = keep_g[sl]
        acc_kept = float((pred[k] == yva[k]).mean()) if k.any() else float("nan")
        keep_rate = float(k.mean())
        ans_by_task[t] = {
            "acc_all": acc_all, "acc_kept": acc_kept,
            "keep_rate": keep_rate, "n_kept": int(k.sum()),
        }
        print(f"  {t:10s} acc_all={acc_all:.3f} acc_kept={acc_kept:.3f} "
              f"keep={keep_rate:.3f} ({int(k.sum())}/{len(k)})")

    # ----------------------------- summary json ----------------------------
    summary = {
        "meta": {
            "model": args.size, "model_id": resolve_model(args.size),
            "layer": layer,
            "closed_train": int(len(Xclosed_tr)), "closed_val": int(len(Xclosed_va)),
            "n_open": len(OPEN_PROMPTS), "n_open_fit": len(Xopen_fit),
            "n_open_eval": len(Xopen_eval),
            "open_sample": OPEN_PROMPTS[:3],
        },
        "gate_geometric": {
            "closed_train_covered_mean": float(cov_closed_tr.mean()),
            "closed_val_covered_mean": float(cov_closed.mean()),
            "open_eval_covered_mean": float(cov_open.mean()),
            "anchored_thresholds": {f"p{q}": v for q, v in geo_thresh.items()},
        },
        "gate_trained_train": {
            "closed_misclassified_as_open": float((p_closed_tr >= 0.5).mean()),
            "open_misclassified_as_closed": float((p_open_tr < 0.5).mean()),
        },
        "answer_when_kept_geometric@0.7": ans_by_task,
    }
    out = args.out or os.path.join(
        RESULTS_DIR, f"gate_ar_{model_slug(args.size)}_t{args.max_train}_v{args.max_val}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[wrote] {out}")

    print("\n========== HEADLINE ==========")
    print(f"closed train cov mean: {cov_closed_tr.mean():.3f}"
          f" | closed val: {cov_closed.mean():.3f}"
          f" | open eval: {cov_open.mean():.3f}")
    for q, T in geo_thresh.items():
        print(f"geometric ≥ closed-train p{q} ({T:.0f}):"
              f" closed-keep={float((cov_closed>=T).mean()):.3f}"
              f" open-escalate={float((cov_open<T).mean()):.3f}")
    print("trained logistic @0.5:"
          f" closed-keep={float((p_closed<0.5).mean()):.3f}"
          f" open-escalate={float((p_open>=0.5).mean()):.3f}"
          f" [mean p_open on eval={p_open.mean():.3f}]")
    print("==============================")


if __name__ == "__main__":
    main()