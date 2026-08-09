import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import ALLOWED_SIZE, default_layers, load_model

SUBJECTS = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank"]
REL = "is taller than"


def build_dataset(n_per, seed=0, min_hops=2, n_distract=2,
                  balance_hops=True, max_hops=7):
    """Rows require multi-hop transitive closure; facts shuffled + distractor facts.
    label=1 iff the query pair is consistent with the true hidden ranking."""
    rng = random.Random(seed)
    n_hops = max_hops - min_hops + 1
    rows = []
    while len(rows) < n_per:
        order = rng.sample(SUBJECTS, len(SUBJECTS))
        if balance_hops:
            d = rng.randint(min_hops, max_hops)
        else:
            i = rng.randint(0, len(SUBJECTS) - min_hops - 1)
            j = rng.randint(i + min_hops, len(SUBJECTS) - 1)
            d = j - i
        i = rng.randint(0, len(SUBJECTS) - d - 1)
        j = i + d
        a, b = order[i], order[j]
        facts = [f"{order[k]} {REL} {order[k+1]}." for k in range(i, j)]
        path_edges = set(range(i, j))
        extras = [k for k in range(len(SUBJECTS) - 1) if k not in path_edges]
        rng.shuffle(extras)
        facts += [f"{order[k]} {REL} {order[k+1]}." for k in extras[:n_distract]]
        rng.shuffle(facts)
        if rng.random() < 0.5:
            query = f"Is {a} taller than {b}?"
            label = 1
        else:
            query = f"Is {b} taller than {a}?"
            label = 0
        prompt = " ".join(facts) + " " + query
        # track ground-truth shortest hop distance between a and b in ranking
        rows.append({"prompt": prompt, "label": label, "hops": d})
    return rows


def surface_oracle(prompt):
    """Cheap surface baseline: does the query subject (2 mots before 'taller')
    appear before the object in the raw prompt text? A probe beating this shows it
    captures non-surface structure."""
    q = prompt.split("Is ")[-1]
    inner = q[:-1]  # strip '?'
    if inner.endswith("?"):
        inner = inner[:-1]
    subj, _, obj = inner.partition(" is taller than ")
    before = prompt.find(subj)
    after = prompt.rfind(obj)
    return 1 if before < after else 0


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


def _decode(prompt, tokenizer, model):
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=4, do_sample=False,
                         temperature=None, top_p=None)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).lower()


def lm_head_readout(tokenizer, model, rows, style="zero"):
    """The decoder as readout: greedy decode and class 1 if 'yes' before 'no'
    in first generated tokens. `style` controls the conditioning budget so the
    comparison is fairer than a bare zero-shot head:
      - 'zero': bare appended instruction (the original, non-matched baseline)
      - 'few' : fixed few-shot demos using subjects outside the dataset, so the
                decoder sees the answer format the same way the probe 'sees'
                the task through its training labels.
    """
    demo = ("Wolf is taller than Xena. Xena is taller than Yara. "
            "Is Wolf taller than Yara? yes\n"
            "Leon is taller than Noah. Noah is taller than Owen. "
            "Is Owen taller than Leon? no\n")
    vals = []
    for r in rows:
        if style == "few":
            p = demo + r["prompt"] + " Answer with 'yes' or 'no':"
        else:
            p = r["prompt"] + " Answer with 'yes' or 'no':"
        vals.append(1 if "yes" in _decode(p, tokenizer, model) else 0)
    return np.array(vals)


def random_projection_probe(X_tr, y_tr, X_va, y_va, dims=(16, 64, 256), n_runs=5):
    """Selectivity control: fit the same logistic probe on random Gaussian
    projections of the features. A genuinely linearly-encoded relation should
    survive coarse random directions; a probe that only overfits a fragile
    coordinate alignment should collapse. Reported as mean val acc over runs.
    """
    rng = np.random.default_rng(0)
    mean = X_tr.mean(0)
    Z_tr, Z_va = X_tr - mean, X_va - mean
    out = {}
    for d in dims:
        accs = []
        for _ in range(n_runs):
            P = rng.standard_normal((d, Z_tr.shape[1]))
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(Z_tr @ P.T, y_tr)
            accs.append(accuracy_score(y_va, clf.predict(Z_va @ P.T)))
        out[d] = float(np.mean(accs))
    return out


def mlp_probe(X_tr, y_tr, X_va, y_va, hidden=128, epochs=150, lr=1e-3, seed=0,
              device="cpu", return_net=False, n_classes: int | None = None):
    """Higher-capacity single-pass readout: an MLP on the same last-layer vector.

    This is the 'capacity ceiling' for the loop-vs-surfaces question: if an MLP
    (nonlinear, same frozen input, still ONE forward pass) closes the gap to the
    few-shot decoder, the loop adds nothing the latent cannot hold. If the loop
    still beats it at deep serial depth, the loop is doing irreducible serial
    computation.

    Binary (``n_classes`` is None or ≤2): returns sigmoid probs in [0,1],
    BCEWithLogits, single logit. Multi-class (``n_classes`` > 2): returns
    argmax class indices (int array), CrossEntropy, ``n_classes`` logits.
    Fitted nets stash ``_probe_mu`` / ``_probe_sd`` / ``_n_classes``.

    `device` pins the (single-pass MLP) fit to cuda/mps/cpu -- the probe is the
    stage that dominates wall-time on wide d_model, so it belongs on the GPU."""
    torch.manual_seed(seed)
    d = X_tr.shape[1]
    y_arr = np.asarray(y_tr)
    if n_classes is None:
        # Infer: bool / 0-1 float → binary; integer labels → max+1.
        if y_arr.dtype == bool or set(np.unique(y_arr.astype(float))).issubset({0.0, 1.0}):
            n_classes = 2
        else:
            n_classes = int(np.max(y_arr)) + 1
    n_classes = int(n_classes)
    multi = n_classes > 2

    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
    xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32, device=device)
    xv = torch.tensor((X_va - mu) / sd, dtype=torch.float32, device=device)
    out_dim = n_classes if multi else 1
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                        nn.Linear(hidden, out_dim)).to(device)
    # Stash the standardization constants on the module so a saved head is
    # self-describing (a reviewer needs mu/sd to reproduce predictions).
    net._probe_mu, net._probe_sd = mu, sd
    net._n_classes = n_classes
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if multi:
        yt = torch.tensor(y_arr.astype(np.int64), dtype=torch.long, device=device)
        lossf = nn.CrossEntropyLoss()
    else:
        yt = torch.tensor(y_arr.astype(np.float32), dtype=torch.float32,
                          device=device).unsqueeze(1)
        lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = net(xt)
        loss = lossf(logits, yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        out = net(xv).detach().cpu().numpy()
    if multi:
        p = out.argmax(axis=-1).astype(np.int64)
    else:
        logits = out.squeeze(-1)
        p = torch.sigmoid(torch.tensor(logits)).numpy()
    if return_net:
        return p, net, mu, sd
    return p


def _default_layers(n):
    return default_layers(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=sorted(ALLOWED_SIZE))
    ap.add_argument("--per-depth", type=int, default=300)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--no-lm", action="store_true", help="skip LM-head readout (slow)")
    ap.add_argument("--mlp", action="store_true",
                    help="run higher-capacity MLP ceiling probe on the last layer")
    ap.add_argument("--no-balance-hops", action="store_true",
                    help="use historical skewed hop sampling")
    args = ap.parse_args()
    model_name = f"Qwen/Qwen3-{args.size}"
    n_layers = ALLOWED_SIZE[args.size]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else _default_layers(n_layers))

    model, tok = load_model(args.size)

    BH = not args.no_balance_hops
    train = build_dataset(args.per_depth, seed=0, balance_hops=BH)
    val = build_dataset(args.per_depth // 4, seed=1, balance_hops=BH)

    for split, rows in (("train", train), ("val", val)):
        data = extract(tok, model, rows, layers)
        np.savez(f"/tmp/h_{split}.npz", **data,
                 labels=np.array([r["label"] for r in rows]),
                 hops=np.array([r["hops"] for r in rows]))
        print(f"{split}: {len(rows)} rows")

    tr, va = np.load("/tmp/h_train.npz"), np.load("/tmp/h_val.npz")
    y_tr, y_va = tr["labels"], va["labels"]

    print("\n=== Surface oracle (mention-order baseline) ===")
    print(f"  val {accuracy_score(y_va, [surface_oracle(r['prompt']) for r in val]):.3f}")

    print("\n=== Linear probe accuracy per layer ===")
    for k in tr.files:
        if not k.startswith("layer"):
            continue
        X_tr, X_va = tr[k], va[k]
        mu = X_tr.mean(0)
        X_tr, X_va = X_tr - mu, X_va - mu
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_va, clf.predict(X_va))
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        print(f"  {k}: train {tr_acc:.3f}  val {acc:.3f}")

    print("\n=== Controls ===")
    # 1. Label-shuffle null: a probe should be chance on shuffled labels.
    last = f"layer{n_layers - 1}"
    X_tr_l = tr[last]
    rng = np.random.default_rng(0)
    c2 = LogisticRegression(C=1.0, max_iter=2000)
    ys = rng.permutation(y_tr)
    c2.fit(X_tr_l - X_tr_l.mean(0), ys)
    print(f"  label-shuffle null ({last}): "
          f"val {accuracy_score(y_va, c2.predict(va[last] - X_tr_l.mean(0))):.3f}")

    # 2. Refit last-layer probe on true labels for the hop-stratified boundary.
    mu = X_tr_l.mean(0)
    X_tr_s, X_va_s = X_tr_l - mu, va[last] - mu
    probe = LogisticRegression(C=1.0, max_iter=2000)
    probe.fit(X_tr_s, y_tr)
    probe_pred = probe.predict(X_va_s)

    print("\n=== Random-projection probe (selectivity control) ===")
    rp = random_projection_probe(X_tr_s, y_tr, X_va_s, y_va)
    for d, a in rp.items():
        print(f"  dim {d:>4}: val {a:.3f}")

    if args.mlp:
        print("\n=== MLP ceiling probe (last layer, single pass) ===")
        mlp_prob = mlp_probe(X_tr_l, y_tr, va[last], y_va)
        mlp_pred = (mlp_prob > 0.5).astype(int)
        print(f"  val {accuracy_score(y_va, mlp_pred):.3f}")
    else:
        mlp_pred = None

    if not args.no_lm:
        print("\n=== LM head readout (decoder as readout) ===")
        pred0 = lm_head_readout(tok, model, val, style="zero")
        print(f"  zero-shot val {accuracy_score(y_va, pred0):.3f}")
        pred = lm_head_readout(tok, model, val, style="few")
        print(f"  few-shot val  {accuracy_score(y_va, pred):.3f}")
    else:
        pred = np.zeros_like(y_va)

    print("\n=== Serial-depth boundary: probe vs decoder by hop count ===")
    hops_va = va["hops"]
    hdr = f"  {'hops':>4} {'n':>4} {'probe':>7} {'mlp':>6} {'decoder':>8} {'shuffle':>8}"
    if mlp_pred is None:
        hdr = f"  {'hops':>4} {'n':>4} {'probe':>7} {'decoder':>8} {'shuffle':>8}"
    print(hdr)
    for h in sorted(set(hops_va.tolist())):
        m = hops_va == h
        if m.sum() == 0:
            continue
        pa = accuracy_score(y_va[m], probe_pred[m])
        da = accuracy_score(y_va[m], pred[m])
        sa = accuracy_score(y_va[m], c2.predict(X_va_s[m]))
        if mlp_pred is not None:
            ma = accuracy_score(y_va[m], mlp_pred[m])
            print(f"  {h:>4} {m.sum():>4} {pa:>7.3f} {ma:>6.3f} {da:>8.3f} {sa:>8.3f}")
        else:
            print(f"  {h:>4} {m.sum():>4} {pa:>7.3f} {da:>8.3f} {sa:>8.3f}")


if __name__ == "__main__":
    main()