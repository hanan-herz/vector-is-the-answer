"""Serial-composition boundary test.

Task frame: literals relating successive values through a rule, with the answer
requiring a K-step chain. Repeated relabeling like "A>B>..>" is the PARALLEL
(one-pass) case the probe wins. Here the rule is a VALUE TRANSFORM -- applying
it K times -- whose output is not recoverable from any single step alone unless
the model composes the chain. This is the cell where the thesis PREDICTS the
single-pass readout may collapse and the autoregressive loop (which emits each
intermediate) may win -- the boundary the paper claims to be hunting.

Concretely: rule "x -> x + k (mod p)" is too cheap (additive, linear). Use a
genuinely serial map over a small ring where composition does not collapse to a
first-pass sum:

    f(x) = (a*x + b) mod p,   with a !~ {0,1}, so f^K is a nontrivial chain.

The premise gives s and K and the cofficient `a`/`b`; the readout must carry the
composition result f^K(s). Label = parity of f^K(s) (binary). We sweep K so the
'depth' is a controlled variable; we then compare single-pass probes with the
few-shot autoregressive loop, per K.

Run:  python boundary.py --size 0.6B
"""
import argparse
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import ALLOWED_SIZE, load_model
from probe_main import mlp_probe, lm_head_readout


def rule(x, a, b, p):
    return (a * x + b) % p


def make_rows(n, seed=0, p=11, min_K=2, max_K=6, n_per_depth=None):
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        K = rng.randint(min_K, max_K)          # serial depth (the variable)
        a = rng.choice([k for k in range(2, p) if k != 1])  # nontrivial mult
        b = rng.randint(0, p - 1)
        s = rng.randint(0, p - 1)
        x = s
        for _ in range(K):
            x = rule(x, a, b, p)
        rows.append({"prompt": f"start {s}: step 1 of {K}, rule (+{b}, x{a}). "
                     f"final value parity?",
                     "K": K, "label": x % 2})
    return rows


def surface_oracle(r):
    s = int(r["prompt"].split("start ")[1].split(":")[0])
    return s % 2


def loop_readout(tokenizer, model, rows):
    """The autoregressive loop, given an explicit step-by-step instruction and
    a correct worked example, verifies by composing the map. If the loop wins
    where the single-pass probe collapses, that is the boundary the thesis
    predicts; if it fails too, the base model simply lacks the knowledge (the
    task is not discriminating latent-vs-loop)."""
    demo = ("Work out the final value in steps, then answer only YES if it is odd "
            "or NO if it is even.\n"
            "start 2: the map is x -> (2x+1) mod 5, apply 3 times.\n"
            "  2 -> 0 -> 1 -> 3. final 3, odd. YES.\n"
            "start 1: the map is x -> (3x) mod 7, apply 2 times.\n"
            "  1 -> 3 -> 2. final 2, even. NO.\n"
            "Now:\n")
    preds = []
    for r in rows:
        p = demo + r["prompt"] + "\nfinal value parity answer:"
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                             temperature=None, top_p=None)
        tail = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True).upper()
        ly, ln = tail.rfind("YES"), tail.rfind("NO")
        preds.append(1 if ly > ln else 0)
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=sorted(ALLOWED_SIZE))
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--no-lm", action="store_true")
    args = ap.parse_args()

    n_layers = ALLOWED_SIZE[args.size]
    model, tok = load_model(args.size)
    train = make_rows(args.n_train, seed=0)
    val = make_rows(args.n_val, seed=1)
    y_tr = np.array([r["label"] for r in train])
    y_va = np.array([r["label"] for r in val])

    def collect(rows):
        buf = []
        h = model.model.layers[n_layers - 1].register_forward_hook(
            lambda m, i, o: buf.append((o[0] if isinstance(o, tuple)
                                        else o).detach().float().cpu()))
        for r in rows:
            ids = tok(r["prompt"], return_tensors="pt").to(model.device)
            model(**ids)
        h.remove()
        return torch.cat([b[:, -1, :] for b in buf], 0).numpy()

    X_tr, X_va = collect(train), collect(val)

    print(f"\n=== surface oracle (val) {accuracy_score(y_va, [surface_oracle(r) for r in val]):.3f} ===")

    mu = X_tr.mean(0)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_tr - mu, y_tr)
    lin = accuracy_score(y_va, clf.predict(X_va - mu))
    mlp = float((mlp_probe(X_tr, y_tr, X_va, y_va) > 0.5).mean())
    print(f"=== single-pass readouts ===")
    print(f"  linear (last token)  {lin:.3f}")
    print(f"  mlp   (last token)   {mlp:.3f}")

    rng = np.random.default_rng(0)
    null = LogisticRegression(C=1.0, max_iter=2000)
    null.fit(X_tr - mu, rng.permutation(y_tr))
    print(f"  label-shuffle null   {accuracy_score(y_va, null.predict(X_va - mu)):.3f}")

    if not args.no_lm:
        dec = accuracy_score(y_va, loop_readout(tok, model, val))
        print(f"=== loop (explicit step-by-step, few-shot) {dec:.3f} ===")

    print(f"\n=== by serial depth K ===")
    print(f"  {'K':>2} {'n':>4} {'linear':>7} {'mlp':>7} {'null':>7} {'dec':>7}")
    Ks = sorted(set(r["K"] for r in val))
    for k in Ks:
        m = np.array([r["K"] == k for r in val])
        print(f"  {k:>2} {m.sum():>4} "
              f"{accuracy_score(y_va[m], clf.predict(X_va[m]-mu)):>7.3f} "
              f"{accuracy_score(y_va[m], (mlp_probe(X_tr, y_tr, X_va[m], y_va[m]) > 0.5)):>7.3f} "
              f"{accuracy_score(y_va[m], null.predict(X_va[m]-mu)):>7.3f}",
              end="")
        if not args.no_lm:
            d = np.array([r["K"] == k for r in val])
            acc = accuracy_score(y_va[d], loop_readout(tok, model, [r for r in val if r["K"] == k]))
            print(f" {acc:>7.3f}", end="")
        print()


if __name__ == "__main__":
    main()