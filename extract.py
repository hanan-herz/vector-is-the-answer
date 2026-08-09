"""Closed-set relation extraction + slot-filling from a frozen forward pass.

The seed's useful-claim, made concrete: structured text extraction does not need
token generation when the output target is a *defined set* (a schema). Two win
probes, both on Qwen3 residual vectors, ONE forward pass:

  A) RELATION classification -- sentence (subject, object) pair -> which of K
     closed relations (born_in/works_at/founded/located_in).
  B) OBJECT slot-fill -- TRUNCATED prompt "(subject) {relword}" with the object
     REMOVED -> predict which closed-set object fills the slot.
     (Design note: if the object were present it would BE the final token and
     reading it would be trivial. Truncation forces the head to recover the
     (subject, relation) -> object binding from the latent, not the surface.
     This is the "predict the field that isn't in the prompt" extraction.)

Controls (the deconfound suite, per the program's discipline):
  - surface bag-of-words baseline (string features, no residual) as the lower
    bound a shallow lexical readout gives
  - label-shuffle null
  - random-projection selectivity
  - REWORD transfer: same relations written with a held-out set of rel-words
    never seen in training; a head that memorized the template collapses, a
    semantic head transfers.

Run:  python extract.py --size 0.6B
"""
import argparse
import random
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import ALLOWED_SIZE, default_layers, load_model
from probe_main import random_projection_probe

PEOPLE = ["Alice", "Bob", "Carol", "David", "Eve", "Fiona", "Grant", "Helen"]
COMPANIES = ["Nimbus", "Orbital", "Verdant", "Halcyon", "Lucent", "Quanta", "Solara", "Marble"]
CITIES = ["Berlin", "Oslo", "Lima", "Osaka", "Lyon", "Cadiz", "Minsk", "Perth"]

# relation -> (subject-role vocab, object-role vocab, relword pool)
# relwords: idx 0,1 are TRAIN-pool; idx 2,3 are held-out for the REWORD check.
RELATIONS = {
    "born_in":    (PEOPLE, CITIES,    ["was born in", "hails from", "comes from", "grew up in"]),
    "works_at":   (PEOPLE, COMPANIES, ["works at", "is employed at", "holds a post at", "is part of"]),
    "founded":    (PEOPLE, COMPANIES, ["founded", "established", "created", "launched"]),
    "located_in": (COMPANIES, CITIES, ["is located in", "is situated in", "can be found in", "is based in"]),
}
REL_LIST = sorted(RELATIONS)
REL_INDEX = {r: i for i, r in enumerate(REL_LIST)}
TRAIN_IDX, REWORD_IDX = (0, 1), (2, 3)


def build_rows(n, seed=0, rel_idx=TRAIN_IDX, force_rel=None):
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        rel = force_rel if force_rel else rng.choice(REL_LIST)
        ppl, obj, pool = RELATIONS[rel]
        s = rng.choice(ppl)
        # DETERMINISTIC (subject, relation) -> object: a KB triple. Object is a
        # fixed function of a subject-index and relation so the head CAN recover
        # the binding; the reword split then tests generalization across
        # wording, not memorization of random noise.
        s_idx = (ppl.index(s) + REL_INDEX[rel] * 5) % len(obj)
        o = obj[s_idx]
        rw = pool[rng.choice(rel_idx)]
        rows.append({
            "prompt": f"{s} {rw} {o}.",
            "trunc": f"{s} {rw}",           # object REMOVED for slot-fill
            "relation": rel,
            "subject": s,
            "object": o,
            "relword": rw,
        })
    return rows


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


@torch.no_grad()
def extract_last(tokenizer, model, rows, layer):
    buf = []
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: buf.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
    for r in rows:
        ids = tokenizer(r["prompt"], return_tensors="pt").to(model.device)
        model(**ids)
    h.remove()
    return torch.cat([b[:, -1, :] for b in buf], 0).numpy()


def surface_bow_feats(rows, vocab_by_word):
    """Bag-of-words features over a FIXED shared vocab (string-level surface)."""
    X = np.zeros((len(rows), len(vocab_by_word)))
    for j, r in enumerate(rows):
        for w in re.findall(r"[a-z]+", r["prompt"].lower()):
            i = vocab_by_word.get(w)
            if i is not None:
                X[j, i] += 1
    return X


def surface_vocab(all_rows):
    stop = {"the", "a", "an", "and", "of", "to"}
    words = set()
    for r in all_rows:
        words.update(re.findall(r"[a-z]+", r["prompt"].lower()))
    vocab = sorted(words - stop)
    return {w: i for i, w in enumerate(vocab)}


def mlp_mc(X_tr, y_tr, X_va, y_va, hidden=128, epochs=200, lr=1e-3, seeds=(0, 1, 2, 3, 4)):
    """Multi-class MLP readout, seed-swept (mean, std over seeds)."""
    accs = []
    n_cls = int(max(int(np.max(y_tr)), int(np.max(y_va))) + 1)
    for seed in seeds:
        torch.manual_seed(seed)
        mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
        xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32)
        xv = torch.tensor((X_va - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y_tr, dtype=torch.long)
        net = nn.Sequential(nn.Linear(X_tr.shape[1], hidden), nn.ReLU(),
                            nn.Linear(hidden, n_cls))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(net(xt), yt)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(xv).argmax(-1).numpy()
        accs.append(accuracy_score(y_va, pred))
    return float(np.mean(accs)), float(np.std(accs))


@torch.no_grad()
def decoder_relation(tokenizer, model, rows):
    """The autoregressive loop as the readout: greedy few-shot extraction.
    Compares the one-pass head against the loop that emits tokens."""
    demo = ("Alice was born in Berlin.  relation: born_in\n"
            "Bob founded Nimbus.          relation: founded\n")
    preds = []
    for r in rows:
        p = demo + r["prompt"] + " relation:"
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=6, do_sample=False,
                             temperature=None, top_p=None)
        tail = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True).lower()
        lab = REL_INDEX
        hit = next((k for k in REL_LIST if k in tail and tail.index(k) >= 0), None)
        preds.append(REL_INDEX[hit] if hit else REL_INDEX["located_in"])
    return np.array(preds)


@torch.no_grad()
def decoder_slot(tokenizer, model, rows):
    demo = ("Alice works at Nimbus.\nBob founded Orbital.\n"
            "Question: what company? answer: Nimbus / the company is Orbital.\n")
    preds = []
    for r in rows:
        p = demo + r["trunc"] + ". What company? answer:"
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=6, do_sample=False,
                             temperature=None, top_p=None)
        tail = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        hit = next((o for o in COMPANIES if o.lower() in tail.lower()), None)
        obj_order = sorted(COMPANIES)
        preds.append(obj_order.index(hit) if hit else 0)
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=sorted(ALLOWED_SIZE))
    ap.add_argument("--n-train", type=int, default=700)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--decoder", action="store_true",
                    help="also run the autoregressive (few-shot) loop as a reference")
    args = ap.parse_args()

    n_layers = ALLOWED_SIZE[args.size]
    layers = default_layers(n_layers)

    train = build_rows(args.n_train, seed=0)                      # TRAIN relwords
    val_in = build_rows(args.n_val, seed=1)                       # TRAIN wording, fresh pairs
    val_re = build_rows(args.n_val, seed=2, rel_idx=REWORD_IDX)   # held-out wording

    print(f"\n=== A) RELATION classification ({args.size}) ===")
    print(f"  predict which of {len(REL_LIST)} closed relations from the final-token"
          f" residual, ONE forward pass, no tokens")

    model, tok = load_model(args.size)
    layers = default_layers(ALLOWED_SIZE[args.size])

    def extract_layers(rows, key="prompt", only=None):
        """Per-layer {layer: (n, d)} final-token residuals for a text field."""
        use = only or layers
        def pick(i):
            return i in layers if only is None else True
        out = {i: [] for i in use}
        handles = []
        for i in use:
            handles.append(model.model.layers[i].register_forward_hook(
                (lambda j: lambda m, inp, o: out[j].append(
                    (o[0] if isinstance(o, tuple) else o).float().detach()[:, -1, :]))(i)))
        for r in rows:
            ids = tok(r[key], return_tensors="pt").to(model.device)
            model(**ids)
        for h in handles:
            h.remove()
        return {i: torch.cat(v, dim=0).cpu().numpy() for i, v in out.items()}

    y_tr = np.array([REL_INDEX[r["relation"]] for r in train])
    y_in = np.array([REL_INDEX[r["relation"]] for r in val_in])
    y_re = np.array([REL_INDEX[r["relation"]] for r in val_re])

    rng = np.random.default_rng(0)

    sbow = surface_vocab(train)   # TRAIN-only vocab: reword words are unseen
    sb_train, sb_in, sb_re = (surface_bow_feats(train, sbow),
                              surface_bow_feats(val_in, sbow),
                              surface_bow_feats(val_re, sbow))
    clf_s = LogisticRegression(C=1.0, max_iter=3000)
    clf_s.fit(sb_train, y_tr)
    print(f"  surface-BOW (lexical lower bound)  "
          f"in-template {accuracy_score(y_in, clf_s.predict(sb_in)):.3f}   "
          f"reword {accuracy_score(y_re, clf_s.predict(sb_re)):.3f}")

    print(f"  residual linear probe, per layer:")
    Xm = extract_layers(train, "prompt")
    Xin = extract_layers(val_in, "prompt")
    Xre = extract_layers(val_re, "prompt")
    best = None
    for i in layers:
        Xt = Xm[i]; mu = Xt.mean(0)
        clf = LogisticRegression(C=1.0, max_iter=3000)
        clf.fit(Xt - mu, y_tr)
        a_in = accuracy_score(y_in, clf.predict(Xin[i] - mu))
        print(f"    {i}: in-template {a_in:.3f}")
        if best is None or a_in > best[1]:
            best = (i, a_in)
            best_mu = mu
    bl = best[0]
    X_bl_tr, X_bl_in, X_bl_re = Xm[bl], Xin[bl], Xre[bl]
    print(f"  >> best layer {bl} (linear), reporting MLP / controls there:")

    print(f"  residual MLP reword (seed-swept):")
    m_re, s_re = mlp_mc(X_bl_tr, y_tr, X_bl_re, y_re)
    print(f"    {m_re:.3f}+/-{s_re:.3f}  (surface reword {accuracy_score(y_re, clf_s.predict(sb_re)):.3f})")

    clf_n = LogisticRegression(C=1.0, max_iter=3000)
    clf_n.fit(X_bl_tr - best_mu, rng.permutation(y_tr))
    print(f"  label-shuffle null                  "
          f"{accuracy_score(y_in, clf_n.predict(X_bl_in - best_mu)):.3f}")

    if args.decoder:
        print(f"\n  AUTOREGRESSIVE loop (few-shot, emits tokens) reference:")
        d_rel = decoder_relation(tok, model, val_in)
        print(f"    relation chosen (in-template)  {accuracy_score(y_in, d_rel):.3f}")

    print(f"  random-projection selectivity (relation, best layer):")
    rp = random_projection_probe(X_bl_tr - best_mu, y_tr, X_bl_in - best_mu, y_in)
    for d, a in rp.items():
        print(f"    dim {d:>4}: {a:.3f}")

    print(f"\n=== B) OBJECT slot-fill (works_at + founded): input = '(subject)+(relword)', "
          f"object REMOVED ===")
    print(f"  predict which company fills the slot. The object is NOT in the prompt, and it")
    print(f"  depends on BOTH subject and relation, so a surface reader that can only see the")
    print(f"  subject word (relword is novel on the reword split) cannot recover it; only a")
    print(f"  head that reads the relation *semantics* -- which transfers across wording -- can.")

    # person->company relations only, deterministic (subj, rel) -> object
    def slot_rows(n, seed, rel_idx):
        rows = []
        rng = random.Random(seed)
        while len(rows) < n:
            rel = rng.choice(["works_at", "founded"])
            ppl, obj, pool = RELATIONS[rel]
            s = rng.choice(ppl)
            o = obj[(ppl.index(s) + REL_INDEX[rel] * 5) % len(obj)]
            rows.append({"trunc": f"{s} {pool[rng.choice(rel_idx)]}",
                         "subject": s, "object": o, "relation": rel})
        return rows

    slot_tr = slot_rows(args.n_train, seed=3, rel_idx=TRAIN_IDX)
    slot_re = slot_rows(args.n_val, seed=4, rel_idx=REWORD_IDX)
    objs = sorted(COMPANIES)
    o_idx = {o: i for i, o in enumerate(objs)}
    yt = np.array([o_idx[r["object"]] for r in slot_tr])
    yr = np.array([o_idx[r["object"]] for r in slot_re])

    def trunc_at(rows, layer=bl):
        return extract_layers(rows, "trunc", only=[layer])[layer]

    Xt = trunc_at(slot_tr); Xr = trunc_at(slot_re)
    mu_s = Xt.mean(0)
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(Xt - mu_s, yt)
    print(f"  linear slot (reword val)    {accuracy_score(yr, clf.predict(Xr - mu_s)):.3f}   "
          f"(chance {1/len(objs):.3f})")
    clf_s2 = LogisticRegression(C=1.0, max_iter=2000)
    st = surface_bow_feats([{**r, "prompt": r["trunc"]} for r in slot_tr], sbow)
    clf_s2.fit(st, yt)
    s2_val = surface_bow_feats([{**r, "prompt": r["trunc"]} for r in slot_re], sbow)
    print(f"  surface-BOW slot (reword)   {accuracy_score(yr, clf_s2.predict(s2_val)):.3f}   "
          f"(object absent -> chance)")
    clf_n2 = LogisticRegression(C=1.0, max_iter=2000)
    clf_n2.fit(Xt - mu_s, rng.permutation(yt))
    print(f"  label-shuffle null          {accuracy_score(yr, clf_n2.predict(Xr - mu_s)):.3f}")
    m_s, s_s = mlp_mc(Xt, yt, Xr, yr)
    print(f"  MLP slot (seed-swept)       {m_s:.3f}+/-{s_s:.3f}")
    if args.decoder:
        d_slot = decoder_slot(tok, model, slot_re)
        print(f"  AUTOREGRESSIVE loop (few-shot) {accuracy_score(yr, d_slot):.3f}")


if __name__ == "__main__":
    main()