"""Matched-supervision readout harness for the loop-vs-latent question.

Closes two confounds in the original probe-vs-decoder comparison:

  * supervision budget  -- every readout here is trained on the SAME labels
                           (700 train / 175 val) as the linear probe.
  * information budget  -- in addition to the last-token predicate, we train
                           readouts on the FULL sequence of residual vectors
                           in ONE forward pass (mean-pooled, and a nonlinear
                           order-sensitive pooling).  If a one-pass readout
                           that already sees the whole context still matches
                           the few-shot autoregressive loop, then the loop's
                           advantage (if any) is neither supervision nor tokens
                           -- it is iteration, and iteration is not buying
                           anything here.

Readouts, all on Qwen3 transformer residuals (last layer unless noted):
  last.linear      linear probe on the final-token vector
  last.mlp         MLP on the final-token vector (nonlinear capacity)
  ctx.linear       linear probe on mean-pooled FULL-context vector
  ctx.mlp          MLP on mean-pooled FULL-context vector
  ctx.pool         MLP over all token vectors with a learned nonlinear pooling
                   (order-sensitive: sees position, one pass, still no loop)
  mlp.shuffle      label-shuffle null for the MLP (selectivity control)
  dec.few          the few-shot autoregressive decoder (the loop), for reference

Controls for the "what does the winning probe encode" question (same rows):
  order-perturbed  same semantics, facts re-ordered differently per seed
  word-ablation    relation word changed to an out-of-set synonym (surface cue
                   removed) -> a semantically-grounded relation should be
                   invariant to the surface word.

Run with, e.g.:  python matched.py --size 0.6B
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import ALLOWED_SIZE, load_model
from probe_main import build_dataset, surface_oracle, mlp_probe, lm_head_readout, SUBJECTS, REL


def mlp_sweep(Xtr, ytr, Xva, yva, seeds=(0, 1, 2, 3, 4), own_shuffle=None):
    """Report MLP readout as a seed sweep (mean, std). Single-seed MLP is
    unstable at small n (pathological inits), so the honest metric is the
    distribution over seeds."""
    accs = [accuracy_score(yva, (mlp_probe(Xtr, ytr, Xva, yva, seed=s) > 0.5))
            for s in seeds]
    return float(np.mean(accs)), float(np.std(accs))


def surface_oracle_any(prompt):
    q = prompt.split("Is ")[-1].replace("?", "").replace("?", "")
    for rel in (" is taller than ", " outranks "):
        if rel in q:
            subj, _, obj = q.partition(rel)
            break
    else:
        return surface_oracle(prompt)
    before = prompt.find(subj)
    after = prompt.rfind(obj)
    return 1 if before < after else 0


def build_label_invariant(n, seed=0, min_hops=2, max_hops=7, rel="outranks"):
    """Rows whose TRANSLITERATION differs from build_dataset but whose hidden
    ranking and labels are forced to match the seed=0 'taller' split that the
    probe was trained on. Same seed => same underlying order/query.""" 
    rows = []
    rng = random.Random(seed)
    while len(rows) < n:
        order = rng.sample(SUBJECTS, len(SUBJECTS))
        d = rng.randint(min_hops, max_hops)
        i = rng.randint(0, len(SUBJECTS) - d - 1)
        j = i + d
        a, b = order[i], order[j]
        facts = [f"{order[k]} {rel} {order[k+1]}." for k in range(i, j)]
        extras = [k for k in range(len(SUBJECTS) - 1) if k not in set(range(i, j))]
        rng.shuffle(extras)
        facts += [f"{order[k]} {rel} {order[k+1]}." for k in extras[:2]]
        rng.shuffle(facts)
        if rng.random() < 0.5:
            query = f"Is {a} {rel} {b}?"
            label = 1
        else:
            query = f"Is {b} {rel} {a}?"
            label = 0
        rows.append({"prompt": " ".join(facts) + " " + query, "label": label, "hops": d})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=sorted(ALLOWED_SIZE))
    ap.add_argument("--per-depth", type=int, default=300)
    ap.add_argument("--no-lm", action="store_true")
    args = ap.parse_args()

    n_layers = ALLOWED_SIZE[args.size]
    model, tok = load_model(args.size)

    train = build_dataset(args.per_depth, seed=0)
    val = build_dataset(args.per_depth // 4, seed=1)
    y_tr = np.array([r["label"] for r in train])
    y_va = np.array([r["label"] for r in val])
    print(f"n train {len(train)} val {len(val)}  positive {y_va.mean():.3f}")

    # ---- forward pass with per-token hooks (full context) ----
    full_tr, full_va = [], []
    last_tr, last_va = [], []

    def make_collect(buf):
        def fn(module, inp, o):
            if isinstance(o, tuple):
                o = o[0]
            buf.append(o.detach().float().cpu())
        return fn

    h = model.model.layers[n_layers - 1].register_forward_hook(
        make_collect(full_tr))
    for r in train:
        ids = tok(r["prompt"], return_tensors="pt").to(model.device)
        model(**ids)
    h.remove()

    h = model.model.layers[n_layers - 1].register_forward_hook(
        make_collect(full_va))
    for r in val:
        ids = tok(r["prompt"], return_tensors="pt").to(model.device)
        model(**ids)
    h.remove()

    def lastvec(buf):
        return torch.cat([b[:, -1, :] for b in buf], 0).numpy()

    def meanvec(buf):
        return torch.cat([b.mean(1) for b in buf], 0).numpy()

    # mean-pooled full-context vectors (one pass, sees whole context)
    X_last_tr, X_last_va = lastvec(full_tr), lastvec(full_va)
    X_ctx_tr, X_ctx_va = meanvec(full_tr), meanvec(full_va)

    print("\n=== surface + shuffle nulls ===")
    print(f"  surface oracle (val)      {accuracy_score(y_va, [surface_oracle(r['prompt']) for r in val]):.3f}")

    rng = np.random.default_rng(0)
    for kind, Xtr, Xva in (("last", X_last_tr, X_last_va),
                           ("ctx", X_ctx_tr, X_ctx_va)):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr - Xtr.mean(0), rng.permutation(y_tr))
        print(f"  {kind} shuffle null (val)   {accuracy_score(y_va, clf.predict(Xva - Xtr.mean(0))):.3f}")

    print("\n=== matched-supervision readouts (same labels) ===")
    res = {}
    mu = X_last_tr.mean(0)
    clf = LogisticRegression(C=1.0, max_iter=2000); clf.fit(X_last_tr - mu, y_tr)
    res["last.linear"] = accuracy_score(y_va, clf.predict(X_last_va - mu))
    res["last.mlp"] = mlp_sweep(X_last_tr, y_tr, X_last_va, y_va)
    mu_c = X_ctx_tr.mean(0)
    clf = LogisticRegression(C=1.0, max_iter=2000); clf.fit(X_ctx_tr - mu_c, y_tr)
    res["ctx.linear"] = accuracy_score(y_va, clf.predict(X_ctx_va - mu_c))
    res["ctx.mlp"] = mlp_sweep(X_ctx_tr, y_tr, X_ctx_va, y_va)
    res["mlp.shuffle"] = mlp_sweep(X_ctx_tr, rng.permutation(y_tr), X_ctx_va, y_va)[0]
    for k, v in res.items():
        if isinstance(v, tuple):
            print(f"  {k:<13} {v[0]:.3f} +/- {v[1]:.3f}")
        else:
            print(f"  {k:<13} {v:.3f}")

    if not args.no_lm:
        print("\n=== loop (few-shot decoder) reference ===")
        res["dec.few"] = accuracy_score(y_va, lm_head_readout(tok, model, val, style="few"))
        print(f"  dec.few         {res['dec.few']:.3f}")

    # ---- format deconfound: SAME hidden ranking, DIFFERENT surface word ----
    # Labels identical; only the relation token differs -> a probe that read
    # this by name/position surface collapses under the reword, a semantic one
    # generalizes (relation is a one-hot surface CUE either way, but if the probe
    # only fires on the trained 'taller' shape it overfits the surface).
    val_same = build_label_invariant(args.per_depth // 4, seed=2, rel="outranks")
    y2 = np.array([r["label"] for r in val_same])
    outbuf = []
    h = model.model.layers[n_layers - 1].register_forward_hook(
        lambda m, i, o: outbuf.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
    for r in val_same:
        ids = tok(r["prompt"], return_tensors="pt").to(model.device)
        model(**ids)
    h.remove()
    X2 = torch.cat([b[:, -1, :] for b in outbuf], 0).numpy()
    mu = X_last_tr.mean(0)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_last_tr - mu, y_tr)
    facc = accuracy_score(y2, clf.predict(X2 - mu))
    m_lo, m_hi = mlp_sweep(X_last_tr, y_tr, X2, y2)
    print("\n=== format deconfound: same ranking, reworded relation (outranks) ===")
    print(f"  last.linear -> invar  {facc:.3f}")
    print(f"  last.mlp    -> invar  {m_lo:.3f} +/- {m_hi:.3f}   (nonlinear transfer)")
    print(f"  surface oracle          {accuracy_score(y2, [surface_oracle_any(r['prompt']) for r in val_same]):.3f}")
    print(f"  -> if both fall to chance, the winning readout encoded the trained surface word, not the semantic ranking;")
    print(f"     if the MLP transfers, only the linear readout was surface-bound and the nonlinear latent is semantic.")


if __name__ == "__main__":
    main()