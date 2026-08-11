# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "datasets", "scikit-learn", "numpy", "accelerate", "hf_transfer"]
# ///
"""Local discriminator for the Granite dead-readout finding (Ext 17).

Remote B200 runs show granite-3.1-8b-base with a healthy 8-shot loop
(BoolQ 0.864) but a base-rate residual readout at every layer (BoolQ 0.64,
RuleTaker 0.51), with probes == shuffled-label controls. Cached vectors are
alive (no NaN/dead dims) but no probe variant (standardize / L2 / clip / PCA)
recovers signal.

Hypothesis: the base model only assembles a linearly-decodable verdict under
few-shot context; the 0-shot prompt leaves the answer position uncommitted.

Protocol: replicate bench.py extraction EXACTLY (same fmt_example prompt,
same attention-mask last-token indexing, same StandardScaler+LR(C=1) probe)
on a small BoolQ sample, at two layers, under two prompt regimes:
  0-shot: "Passage: ...\nQuestion: ...\nAnswer:"            (readout regime)
  8-shot: balanced exemplar block + "\n\n" + same stem      (loop regime)
If 8-shot probes jump to ~0.8 while 0-shot stays at base rate, the residual
readout is prompt-regime-dependent for this base model (a finding), not a
harness bug.

Runs fully local (M-series, MPS, bf16). ~16 GB weight download on first run.
"""
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-3.1-8b-base"
LAYERS = (20, 39)
N_TR, N_VA = 800, 800
SEED = 42


def fmt_example(r):
    p = (r.get("passage") or "").strip()
    if p:
        return f"Passage: {p}\nQuestion: {r['question']}\nAnswer:"
    return f"Question: {r['question']}\nAnswer:"


def balanced_fewshot(train, k=8):
    by_c = {0: [], 1: []}
    for r in train:
        by_c[int(r["answer"])].append(r)
    picked, caps = [], {0: 0, 1: 0}
    while len(picked) < k:
        for c in (0, 1):
            if len(picked) >= k:
                break
            if caps[c] < len(by_c[c]):
                picked.append(by_c[c][caps[c]])
                caps[c] += 1
    word = lambda r: "Yes" if r["answer"] else "No"
    return "\n\n".join(fmt_example(r) + " " + word(r) for r in picked)


@torch.no_grad()
def extract(model, tok, texts, layers, batch, max_len, trunc_side):
    tok.truncation_side = trunc_side
    out = {L: [] for L in layers}
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    for b0 in range(0, len(order), batch):
        sub = [texts[i] for i in order[b0:b0 + batch]]
        enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        buf = {}

        def mk(L):
            def hook(m, i, o):
                if isinstance(o, tuple):
                    o = o[0]
                buf[L] = o.flatten(2).detach().float().cpu()
            return hook

        hs = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]
        model(**enc)
        for h in hs:
            h.remove()
        am = enc["attention_mask"].cpu()
        # padding-side-robust last-real-token index (left OR right pad)
        lens = am.shape[1] - 1 - am.flip(1).argmax(1)
        for L in layers:
            out[L].append(buf[L][torch.arange(len(sub)), lens])
        if (b0 // batch) % 10 == 0:
            print(f"    {b0 + len(sub)}/{len(texts)}", flush=True)
    return {L: torch.cat(v, 0).numpy() for L, v in out.items()}


def probe(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=20000).fit(sc.transform(Xtr), ytr)
    return accuracy_score(yva, clf.predict(sc.transform(Xva)))


def main():
    t0 = time.time()
    ds = load_dataset("google/boolq")
    rng = np.random.default_rng(SEED)
    tr_idx = rng.permutation(len(ds["train"]))[:N_TR]
    va_idx = rng.permutation(len(ds["validation"]))[:N_VA]
    train = [dict(ds["train"][int(i)]) for i in tr_idx]
    val = [dict(ds["validation"][int(i)]) for i in va_idx]
    ytr = np.array([int(r["answer"]) for r in train])
    yva = np.array([int(r["answer"]) for r in val])

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("mps").eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    block = balanced_fewshot(train, 8)
    regimes = {
        "0-shot": ([fmt_example(r) for r in train],
                   [fmt_example(r) for r in val], 8, 384, "right"),
    }
    for name, (ttr, tva, bs, ml, side) in regimes.items():
        print(f"[{name}] extracting train ({len(ttr)})", flush=True)
        Xtr = extract(model, tok, ttr, LAYERS, bs, ml, side)
        print(f"[{name}] extracting val ({len(tva)})", flush=True)
        Xva = extract(model, tok, tva, LAYERS, bs, ml, side)
        for L in LAYERS:
            acc = probe(Xtr[L], ytr, Xva[L], yva)
            acc_l2 = probe(normalize(Xtr[L]), ytr, normalize(Xva[L]), yva)
            norms = np.linalg.norm(Xtr[L], axis=1)
            print(f"  [{name}] L{L}: raw {acc:.3f}  L2 {acc_l2:.3f} "
                  f"(base {max(yva.mean(), 1 - yva.mean()):.3f}, "
                  f"norm p10/p50/p90 {np.percentile(norms,[10,50,90]).round(0)})",
                  flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
