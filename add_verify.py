"""Serial-depth boundary test on arithmetic verification.

Task: "Is A + B == C?" where C is the correct sum (label 1) or a digit-perturbed
sum (label 0). Verifying k-digit addition with carry is genuinely serial (the
answers to position i depend on carries from i-1), unlike the parallelizable
transitivity reading task — so it is the right probe for whether a SINGLE forward
pass can hold the answer, or whether the autoregressive loop is needed.

Readouts on identical inputs:
  - linear probe (CONTROL/DIAGNOSTIC only: pooled number is a digit-length
    confound; read it per digit-stratum below)
  - MLP probe (single pass, last-token residual; the deconfounded readout)
  - decoder few-shot greedy "work it out step by step, end YES/NO" (the loop)
  - label-shuffle null, random-projection control
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import ALLOWED_SIZE, default_layers, load_model
from probe_main import random_projection_probe

import itertools


def build_ds(n, seed=0, lo_digits=6, hi_digits=14, rng=None):
    rng = rng or random.Random(seed)
    rows = []
    lo, hi = 10 ** (lo_digits - 1), 10 ** hi_digits - 1
    while len(rows) < n:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        s = a + b
        if rng.random() < 0.5:
            c = s
            label = 1
        else:
            # perturb one digit of the true sum (avoid re-equal to s, and to a/b)
            cs = [d for d in str(s)]
            i = rng.randrange(len(cs))
            orig = cs[i]
            new = rng.choice([d for d in "0123456789" if d != orig])
            cs[i] = new
            c = int("".join(cs))
            if c == s or c == a or c == b:
                continue
            label = 0
        rows.append({
            "prompt": f"Is {a} + {b} = {c}?",
            "label": label,
            "digits": max(len(str(a)), len(str(b)), len(str(s))),
        })
    return rows


def surface_oracle(r):
    # true-sum digit count vs candidate count is a cheap length cue.
    q = r["prompt"].replace("Is ", "").replace("?", "")
    lhs, cand = q.split(" = ")
    a, b = lhs.split(" + ")
    return 1 if len(str(int(a) + int(b))) == len(cand.strip()) else 0


@torch.no_grad()
def extract(tokenizer, model, rows, layers):
    cache = {f"layer{i}": [] for i in layers}
    handles = []

    def mk(i):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            cache[f"layer{i}"].append(out.float()[:, -1, :])
        return fn

    for i in layers:
        handles.append(model.model.layers[i].register_forward_hook(mk(i)))
    for r in rows:
        ids = tokenizer(r["prompt"], return_tensors="pt").to(model.device)
        model(**ids)
    for h in handles:
        h.remove()
    return {k: torch.cat(v, dim=0).cpu().numpy() for k, v in cache.items()}


def decoder_verify(tokenizer, model, rows):
    demo = ("Work it out step by step, then answer ONLY YES or NO.\n"
            "Question: Is 12 + 15 = 27?\n"
            "12 + 15 = 27, so YES.\n"
            "Question: Is 20 + 10 = 31?\n"
            "20 + 10 = 30, not 31, so NO.\n")
    preds = []
    for r in rows:
        p = f"{demo}Question: {r['prompt'].replace('Is ', '')}\nAnswer:"
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                             temperature=None, top_p=None)
        tail = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True).upper()
        # last explicit verdict wins
        last_yes = tail.rfind("YES")
        last_no = tail.rfind("NO")
        preds.append(1 if last_yes > last_no else 0)
    return np.array(preds)


def mlp_probe(X_tr, y_tr, X_va, y_va, hidden=128, epochs=150, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    d = X_tr.shape[1]
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
    xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32)
    xv = torch.tensor((X_va - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(xv).squeeze(1).numpy()
    return (torch.sigmoid(torch.tensor(logits)).numpy() > 0.5).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4B", choices=sorted(ALLOWED_SIZE))
    ap.add_argument("--n-train", type=int, default=700)
    ap.add_argument("--n-val", type=int, default=175)
    ap.add_argument("--no-lm", action="store_true")
    args = ap.parse_args()

    n_layers = ALLOWED_SIZE[args.size]
    layers = default_layers(n_layers)
    model, tok = load_model(args.size)

    train = build_ds(args.n_train, seed=0)
    val = build_ds(args.n_val, seed=1)

    for split, rows in (("train", train), ("val", val)):
        data = extract(tok, model, rows, layers)
        np.savez(f"/tmp/av_{split}.npz", **data,
                 labels=np.array([r["label"] for r in rows]),
                 digits=np.array([r["digits"] for r in rows]))

    tr, va = np.load("/tmp/av_train.npz"), np.load("/tmp/av_val.npz")
    y_tr, y_va = tr["labels"], va["labels"]

    print(f"\n=== surface oracle (length cue) ===")
    print(f"  val {accuracy_score(y_va, [surface_oracle(r) for r in val]):.3f}")

    last = f"layer{n_layers - 1}"
    X_tr_l, X_va_l = tr[last], va[last]
    mu = X_tr_l.mean(0)
    X_tr_s, X_va_s = X_tr_l - mu, va[last] - mu

    print("\n=== linear probe, per layer ===")
    print("  (CONTROL/DIAGNOSTIC: linear pooled number is a digit-length")
    print("   confound -- see per-digit-stratum breakdown below; treat only")
    print("   the MLP readout as a candidate result)")
    best = None
    for k in tr.files:
        if not k.startswith("layer"):
            continue
        X_tr, X_va = tr[k], va[k]
        m = X_tr.mean(0)
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X_tr - m, y_tr)
        acc = accuracy_score(y_va, clf.predict(X_va - m))
        if best is None or acc > best[1]:
            best = (k, acc)
        print(f"  {k}: val {acc:.3f}")
    print(f"  best {best[0]} {best[1]:.3f}")

    print("\n=== MLP probe (single pass) ===")
    print("  (RECOMMENDED READOUT: nonlinear single-pass, deconfounded)")
    mlp_pred = mlp_probe(X_tr_l, y_tr, va[last], y_va)
    print(f"  val {accuracy_score(y_va, mlp_pred):.3f}")

    print("\n=== controls ===")
    rng = np.random.default_rng(0)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_tr_s, rng.permutation(y_tr))
    print(f"  label-shuffle null: {accuracy_score(y_va, clf.predict(X_va_s)):.3f}")
    print("  random-projection selectivity (linear, add task):")
    rp = random_projection_probe(X_tr_l, y_tr, va[last], y_va)
    for d, a in rp.items():
        print(f"    dim {d:>4}: val {a:.3f}")

    if not args.no_lm:
        print("\n=== decoder (loop) few-shot verification ===")
        pred = decoder_verify(tok, model, val)
        print(f"  val {accuracy_score(y_va, pred):.3f}")
    else:
        pred = np.zeros_like(y_va)

    print("\n=== probe vs decoder by digit-count ===")
    print("  (DECONFOUND CHECK: linear collapses within a fixed digit-length,")
    print("   exposing the length confound; MLP is the real readout)")
    dg = va["digits"]
    print(f"  {'digits':>5} {'n':>4} {'linear':>7} {'mlp':>6} {'decoder':>8}")
    for d in sorted(set(dg.tolist())):
        m = dg == d
        if m.sum() < 5:
            continue
        print(f"  {d:>5} {m.sum():>4} "
              f"{accuracy_score(y_va[m], clf.predict(X_va_s[m])):>7.3f} "
              f"{accuracy_score(y_va[m], mlp_pred[m]):>6.3f} "
              f"{accuracy_score(y_va[m], pred[m]):>8.3f}")


if __name__ == "__main__":
    main()