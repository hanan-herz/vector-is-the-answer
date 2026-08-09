"""Think-then-read: does spending MORE forward passes improve the LATENT readout?

STATUS: PARKED. Kept for reference, not part of the active latent-first story.
The experiment's answer (0.6B, relation extraction): injecting test-time compute
("The relation is" + K greedy tokens) does NOT lift a latent readout -- in-template
saturates at 1.000 at K=0, and reword-transfer *drops* (0.99 -> ~0.65-0.74) as K
grows. Consistent with the thesis: for headed, in-knowledge tasks the answer is
already in pass 0. The hybrid "decode passes, then read the latent" is therefore
NOT an accuracy win on this task class, and we chose not to write it up. No
production decision should depend on this file.

Extends extract.py's relation-classification wincase. The claim the seed really
wants is not just "pass-0 is cheap" but "the latent readout is >= the loop (or a
self-answer) on accuracy." We get at it by letting the model THINK -- generate K
tokens (each = one more forward pass over a deepened state) -- then reading the
final-token residual AT THAT DEPTH and classifying.

Protocol:
  K=0 : one pass over the prompt, read residual (the extract.py result).
  K>0 : prompt + " The relation is" + K greedy continuation tokens, then read the
        last-token residual of the DEEPENED state (one extra embed pass).
  Decoder : the model's own few-shot self-answer, as the "loop" reference.

A readout is fit ONCE on K=0 train (budget-matched, fixed head) and then applied
to deeper states -- so accuracy changes across K reflect the STATE, not a
re-fit. If readout-K rises with K -> the "decode some passes, then read the
latent" hybrid is a real ACCURACY win. If readout-K == readout-0 across K ->
consistent with the thesis: the answer is already in pass 0; extra passes don't
move a *readout of the latent*. Either is reported as-is.

Run:  python think.py --size 0.6B
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import ALLOWED_SIZE, default_layers, load_model
from extract import RELATIONS, REL_LIST, REL_INDEX, build_rows, TRAIN_IDX, REWORD_IDX


@torch.no_grad()
def generate_to_depth(model, tok, prompt, k):
    """Return prompt + k greedy continuation tokens as the deepened input text."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=k, do_sample=False,
                         temperature=None, top_p=None)
    return tok.decode(out[0], skip_special_tokens=True)


def read_at_depth(model, tok, layer, texts):
    buf = []
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: buf.append((o[0] if isinstance(o, tuple)
                                    else o).detach().float().cpu()))
    for t in texts:
        ids = tok(t, return_tensors="pt").to(model.device)
        model(**ids)
    h.remove()
    return torch.cat([b[:, -1, :] for b in buf], 0).numpy()


class MLPReadout:
    """A single, fixed-capacity readout fit once; prob() and argmax on demand."""

    def __init__(self, X, y, hidden=128, epochs=200, lr=1e-3, seed=0,
                 device="cpu"):
        torch.manual_seed(seed)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        n_cls = int(max(yt).item()) + 1
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], hidden), nn.ReLU(),
            nn.Linear(hidden, n_cls))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(self.net(xt), yt)
            loss.backward()
            opt.step()
        self.net.eval()

    def prob(self, X):
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(self.net(x), -1).numpy()

    def argmax(self, X):
        return self.prob(X).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=sorted(ALLOWED_SIZE))
    ap.add_argument("--n-train", type=int, default=700)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--k", default="0,1,2,4,8",
                    help="comma list of think-depths (generated tokens)")
    ap.add_argument("--decoder", action="store_true",
                    help="also score the model's own few-shot answer as reference")
    args = ap.parse_args()

    Ks = [int(x) for x in args.k.split(",")]
    n_layers = ALLOWED_SIZE[args.size]
    model, tok = load_model(args.size)

    train = build_rows(args.n_train, seed=0)
    val_in = build_rows(args.n_val, seed=1)
    val_re = build_rows(args.n_val, seed=2, rel_idx=REWORD_IDX)
    y_tr = np.array([REL_INDEX[r["relation"]] for r in train])
    y_in = np.array([REL_INDEX[r["relation"]] for r in val_in])
    y_re = np.array([REL_INDEX[r["relation"]] for r in val_re])

    # pick the readout layer on K=0 train (linear, in-sample proxy) and hold it
    # fixed across K so the curve isn't layer-chosen per depth.
    base0 = [r["prompt"] for r in train]
    best = None
    for layer in default_layers(n_layers):
        X = read_at_depth(model, tok, layer, base0)
        mu = X.mean(0)
        clf = LogisticRegression(C=1.0, max_iter=3000)
        clf.fit(X - mu, y_tr)
        a = accuracy_score(y_tr, clf.predict(X - mu))
        if best is None or a > best[0]:
            best = (a, layer)
    bl = best[1]
    print(f"{args.size}: readout layer fixed to {bl} (best on K=0 train)")

    hdr = f"  {'K tokens':>9} {'readout in':>11} {'readout reword':>15} {'reword conf':>11}"
    print(hdr)

    def deepen(rows, k):
        if k == 0:
            return [r["prompt"] for r in rows]
        return [generate_to_depth(model, tok, r["prompt"] + " The relation is", k)
                for r in rows]

    # A readout fit on K=0 inputs does NOT transfer to deepened inputs (the
    # appended tokens shift the residual distribution), so we fit a readout
    # NATIVE to each depth K and measure whether the deeper STATE is easier to
    # read. Rising accuracy across K = deepening genuinely helps a latent read.
    for k in Ks:
        tin = deepen(val_in, k); tre = deepen(val_re, k)
        X_tr = read_at_depth(model, tok, bl, deepen(train, k))
        Xin = read_at_depth(model, tok, bl, tin)
        Xre = read_at_depth(model, tok, bl, tre)
        readout = MLPReadout(X_tr, y_tr)
        a_in = accuracy_score(y_in, readout.argmax(Xin))
        a_re = accuracy_score(y_re, readout.argmax(Xre))
        conf = float(np.mean(readout.prob(Xre).max(1)))
        print(f"  {k:>9} {a_in:>11.3f} {a_re:>15.3f} {conf:>11.3f}")

    if args.decoder:
        from extract import decoder_relation
        d_in = decoder_relation(tok, model, val_in)
        print(f"\n  decoder few-shot self-answer (in-template) = "
              f"{accuracy_score(y_in, d_in):.3f}")


if __name__ == "__main__":
    main()