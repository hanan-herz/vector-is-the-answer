# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "datasets", "numpy", "scikit-learn", "accelerate"]
# ///
"""Granite hook-point check: does layers[39]'s hook output equal the tensor
that feeds the final norm + unembedding?

Context: probes on hooked L39 vectors are dead (0.55 < 0.609 base rate) while
the loop (Yes/No logits) is healthy (0.825) on the SAME local load. Logits are
a linear read of norm(h_last), so if the hook truly captured h_last, the
logit direction would be recoverable from the hooked vectors.

Captures, in one forward over 100 BoolQ val prompts:
  A = hook output of model.model.layers[39]      (what bench.py probes)
  B = input to model.model.norm                  (what the unembedding reads)
  logits[:, -1]                                  (the loop's read)
Then: per-row cosine(A, B), accuracy of the logit-diff direction on A and on B
(L2-normalized rows), and a plain LR probe on both for reference.
"""
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-3.1-8b-base"
N = 100


def fmt_example(r):
    return f"Passage: {(r.get('passage') or '').strip()}\nQuestion: {r['question']}\nAnswer:"


@torch.no_grad()
def main():
    ds = load_dataset("google/boolq")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(ds["validation"]))[:N]
    rows = [dict(ds["validation"][int(i)]) for i in idx]
    y = np.array([int(r["answer"]) for r in rows])
    texts = [fmt_example(r) for r in rows]

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("mps").eval()

    cap = {}
    def hook39(m, i, o):
        cap["A"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()
    def hooknorm(m, i, o):
        cap["B"] = i[0].detach().float().cpu()

    h1 = model.model.layers[39].register_forward_hook(hook39)
    h2 = model.model.norm.register_forward_hook(hooknorm)
    A_last, B_last, lg = [], [], []
    for b0 in range(0, N, 8):
        enc = tok(texts[b0:b0 + 8], return_tensors="pt", padding=True,
                  truncation=True, max_length=384).to(model.device)
        out = model(**enc)
        am = enc["attention_mask"].cpu()
        lens = am.shape[1] - 1 - am.flip(1).argmax(1)  # padding-side-robust
        ar = torch.arange(len(lens))
        A_last.append(cap["A"][ar, lens]); B_last.append(cap["B"][ar, lens])
        lg.append(out.logits[:, -1, :].float().cpu())
    h1.remove(); h2.remove()
    A = torch.cat(A_last).numpy(); B = torch.cat(B_last).numpy()
    logits = torch.cat(lg).numpy()

    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
    print(f"cosine(hooked L39, pre-norm h): mean {cos.mean():.4f} min {cos.min():.4f}")

    W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    yes = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no = tok(" No", add_special_tokens=False)["input_ids"][0]
    g = model.model.norm.weight.detach().float().cpu().numpy()
    w = g * (W[yes] - W[no])

    def dir_acc(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return accuracy_score(y, (Xn @ w) > 0)

    def probe_acc(X):
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=20000).fit(sc.transform(X), y)
        return accuracy_score(y, clf.predict(sc.transform(X)))

    print("padding_side:", tok.padding_side)
    d_logit = logits[:, yes] - logits[:, no]
    Xn = A / np.linalg.norm(A, axis=1, keepdims=True)
    d_dir = Xn @ w
    agree = (np.sign(d_logit) == np.sign(d_dir)).mean()
    print(f"sign agreement logit-diff vs Xn@w: {agree:.3f}")
    print("sample d_logit:", d_logit[:4].round(2), " d_dir:", d_dir[:4].round(2))
    # manual logits from captured pre-norm B
    Bt = torch.tensor(B)
    with torch.no_grad():
        h_n = model.model.norm(Bt.to(model.device)).float().cpu().numpy()
    manual = h_n @ W.T
    d_manual = manual[:, yes] - manual[:, no]
    print("manual-vs-actual logit sign agree:",
          (np.sign(d_manual) == np.sign(d_logit)).mean())
    print("norm.weight stats: mean %.3f std %.3f" % (g.mean(), g.std()))
    print("logits_scaling:", getattr(model.config, "logits_scaling", None))
    print("sample manual:", d_manual[:4].round(2), " actual:", d_logit[:4].round(2))
    loop = accuracy_score(y, logits[:, yes] > logits[:, no])
    print(f"loop (logits):            {loop:.3f}")
    print(f"logit-dir on hooked A:    {dir_acc(A):.3f}   probe on A: {probe_acc(A):.3f}")
    print(f"logit-dir on pre-norm B:  {dir_acc(B):.3f}   probe on B: {probe_acc(B):.3f}")
    print(f"scale: stdA {A.std():.2f} stdB {B.std():.2f} "
          f"normA p50 {np.percentile(np.linalg.norm(A,axis=1),50):.0f} "
          f"normB p50 {np.percentile(np.linalg.norm(B,axis=1),50):.0f}")


if __name__ == "__main__":
    main()
