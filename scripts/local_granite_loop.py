# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "datasets", "numpy", "accelerate", "hf_transfer"]
# ///
"""Local Granite loop check — complement to scripts/local_granite_probe.py.

Probes on locally-extracted Granite residual vectors are dead (0.55, below the
0.609 base rate) under BOTH 0-shot and 8-shot prompts. But a linear-separation
argument says that can't coexist with a healthy loop: Yes/No logits are a
*linear* read of the final hidden state (after RMSNorm, i.e. an angular
direction in raw-h space), so a healthy loop forces a separating direction.

This script measures the LOCAL loop (argmax over Yes/No next-token logits,
same scoring convention as bench.py's loop_scores) to discriminate:
  local loop.0 ~ 0.8  -> probe-dead + logits-right on the SAME weights/load:
                         genuine residual-vs-unembedding dissociation
  local loop.0 ~ 0.6  -> local and remote Granite behave differently:
                         remote transformers-image Granite impl suspect
"""
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-3.1-8b-base"
N_VA = 200
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
def loop_acc(model, tok, texts, y, batch, max_len):
    yes_ids = tok(" Yes", add_special_tokens=False)["input_ids"]
    no_ids = tok(" No", add_special_tokens=False)["input_ids"]
    yes_tok, no_tok = yes_ids[0], no_ids[0]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    preds = [0] * len(texts)
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        logits = model(**enc).logits[:, -1, :].float().cpu()
        for j, i in enumerate(idx):
            preds[i] = int(logits[j, yes_tok] > logits[j, no_tok])
        if (b0 // batch) % 10 == 0:
            print(f"    {b0 + len(idx)}/{len(texts)}", flush=True)
    return float(np.mean([p == yy for p, yy in zip(preds, y)]))


def main():
    ds = load_dataset("google/boolq")
    rng = np.random.default_rng(SEED)
    va_idx = rng.permutation(len(ds["validation"]))[:N_VA]
    val = [dict(ds["validation"][int(i)]) for i in va_idx]
    y = [int(r["answer"]) for r in val]
    train = [dict(ds["train"][int(i)]) for i in rng.permutation(len(ds["train"]))[:200]]

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("mps").eval()
    print(f"transformers loop check, {N_VA} val rows", flush=True)

    acc0 = loop_acc(model, tok, [fmt_example(r) for r in val], y, 8, 384)
    print(f"loop.0 (0-shot): {acc0:.3f}  (remote: 0.815)", flush=True)
    block = balanced_fewshot(train, 8)
    acc8 = loop_acc(model, tok, [block + "\n\n" + fmt_example(r) for r in val],
                    y, 2, 2048)
    print(f"loop.8 (8-shot): {acc8:.3f}  (remote: 0.864)", flush=True)


if __name__ == "__main__":
    main()
