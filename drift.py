"""drift.py -- probe: does LOOPING the latent (re-feed the residual) drift
off-manifold?

PARKED. Question raised about the latent-first story: "if we loop the latent,
off-manifold drift risk?" -- measured instead of asserting (Ext 7, RESULTS.md):
bounded norm but real off-manifold + decorrelation, at both 0.6B and 4B. Result
is what justified the "COCONUT must retrain" / "single-pass monitor is the safe
thread" reading; no production decision depends on further work here.

COCONUT-style refeed (arXiv:2412.06769), identity projector = the pessimistic
frozen estimate: take the LAST layer's final-token residual, feed it back as the
next input embedding (dimension equals d_model, so identity works), run one
forward pass, repeat K steps. No sampling, no tokens -- a pure latent loop.

Compare against the model's OWN natural autoregressive trajectory (greedy
argmax tokens) through the SAME metrics. Natural iterates define the on-manifold
baseline; if latent-loop iterates leave it, drift is real.

Metrics per step k:
  norm ||h_k||          -- blows up => divergence
  step ||h_{k+1}-h_k||  -- random-walk growth => no stable fixed point
  off-manifold dist     -- reconstruction error to PCA(P) of REAL pass-0
                           final-token residuals; natural iterates stay low,
                           drift shows it climbing
  cos(h_k, h_0)         -- decorrelation from the start state

Verdict: latent-loop offm/norm staying within natural iterates' band => looping
is stable and the drift worry is over-stated (I'd be wrong); exponential growth
in norm or step, or offm pulling far above the natural band => drift confirmed,
and the §7 latent-loop regime is not a frozen-model option.

Run: python drift.py --size 0.6B
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import ALLOWED_SIZE, load_model
from extract import build_rows


@torch.no_grad()
def last_residual(model, tok, text):
    ids = tok(text, return_tensors="pt").to(model.device)
    h = model(**ids, output_hidden_states=True).hidden_states[-1]
    return h[0, -1].detach().float().cpu()


def manifold_basis(vecs, n_comp):
    """PCA of real residuals; returns (mean, principal axes) on cpu."""
    x = torch.stack(vecs)                                  # [P, d]
    mu = x.mean(0)
    U, S, Vt = torch.linalg.svd((x - mu).double(), full_matrices=False)
    energy = (S ** 2).sum().item()
    kept = (S[:n_comp] ** 2).sum().item() / energy
    return mu, Vt[:n_comp].float(), kept


def off_manifold(x, mu, V):
    proj = (x - mu) @ V.T
    rec = mu + proj @ V
    return (x - rec).norm().item()


@torch.no_grad()
def latent_refeed_iterates(model, x0_vec, steps):
    """COCONUT refeed: residual[d] back in as next embedding, identity proj."""
    its = [x0_vec]
    e = x0_vec.to(model.device).to(model.dtype).reshape(1, 1, -1)
    for _ in range(steps):
        out = model(inputs_embeds=e,
                    attention_mask=torch.ones(1, 1, device=model.device),
                    position_ids=torch.zeros(1, 1, dtype=torch.long,
                                             device=model.device),
                    output_hidden_states=True)
        res = out.hidden_states[-1][0, -1].detach().float().cpu()
        its.append(res)
        e = res.reshape(1, 1, -1).to(model.device).to(model.dtype)
    return torch.stack(its)


@torch.no_grad()
def natural_iterates(model, tok, text, steps):
    """Greedy autoregressive trajectory -- the on-manifold baseline."""
    ids = tok(text, return_tensors="pt").to(model.device)
    its = []
    for _ in range(steps):
        out = model(**ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].detach().float().cpu()
        its.append(h)
        nxt = out.logits[0, -1].argmax().item()
        ids = torch.cat([ids["input_ids"],
                         torch.tensor([[nxt]], device=ids["input_ids"].device)], 1)
        ids = {"input_ids": ids}
    return torch.stack(its)


def metrics(its, mu, V, h0_norm):
    norm = its.norm(2, -1)
    step = torch.cat([torch.zeros(1), (its[1:] - its[:-1]).norm(2, -1)])
    offm = torch.tensor([off_manifold(x, mu, V) for x in its])
    cos0 = F.cosine_similarity(its, its[0:1], -1)
    # normalize step/offm to the natural start norm so units are comparable
    return (norm / h0_norm).numpy(), (step / h0_norm).numpy(), \
           (offm / h0_norm).numpy(), cos0.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=sorted(ALLOWED_SIZE))
    ap.add_argument("--k", type=int, default=10, help="latent-loop refeed steps")
    ap.add_argument("--steps", type=int, default=8, help="natural traj length")
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--pool", type=int, default=200, help="manifold pool rows")
    ap.add_argument("--probe", type=int, default=8, help="probe prompts")
    ap.add_argument("--seed-pool", type=int, default=0)
    ap.add_argument("--seed-probe", type=int, default=7)
    args = ap.parse_args()

    model, tok = load_model(args.size)
    d_model = model.get_input_embeddings().weight.shape[1]
    print(f"{args.size}: d_model={d_model}; refeed uses identity projector "
          f"(last-layer residual -> next input embedding)")

    # manifold pool from REAL pass-0 residuals (seed differs from probe)
    pool_rows = build_rows(args.pool, seed=args.seed_pool)
    pool = [last_residual(model, tok, r["prompt"]) for r in pool_rows]
    mu, V, kept = manifold_basis(pool, args.pca)
    print(f"PCA({args.pca}) of {len(pool)} real residuals keeps "
          f"{kept * 100:.1f}% of variance")

    probe_rows = build_rows(args.probe, seed=args.seed_probe)
    h0 = pool[0].norm(2).item()

    latent_acc = {"norm": [], "step": [], "offm": [], "cos0": []}
    natural_acc = {k: [] for k in latent_acc}
    for r in probe_rows:
        x0 = last_residual(model, tok, r["prompt"])          # start on-manifold
        lits = latent_refeed_iterates(model, x0, args.k)
        nits = natural_iterates(model, tok, r["prompt"], args.steps)
        for name, acc in (("latent", latent_acc), ("natural", natural_acc)):
            n, s, o, c = metrics(lits if name == "latent" else nits, mu, V, h0)
            acc["norm"].append(n); acc["step"].append(s)
            acc["offm"].append(o); acc["cos0"].append(c)

    panel = lambda acc: {k: np.mean(np.vstack(v), 0) for k, v in acc.items()}
    L = panel(latent_acc); N = panel(natural_acc)

    def row(label, ln, nn, unit=""):
        print(f"  {label:<10} {ln:>8.3f} {nn:>8.3f}   |  {unit}")

    K = args.k + 1
    print("\n  norm/||h0|| (relative to a real start state; units of h0)")
    for k in range(K):
        row(f"k={k}", L["norm"][k], N["norm"][k] if k < args.steps else np.nan)

    print("\n  step ||h_{k+1}-h_k||/||h0|| (per-iteration displacement)")
    for k in range(K):
        row(f"k={k}", L["step"][k], N["step"][k] if k < args.steps else np.nan)

    print("\n  off-manifold dist /||h0|| (recon error vs PCA of real residuals)")
    for k in range(K):
        row(f"k={k}", L["offm"][k], N["offm"][k] if k < args.steps else np.nan)

    print("\n  cos(h_k, h_0) (start-state retention)")
    for k in range(K):
        row(f"k={k}", L["cos0"][k], N["cos0"][k] if k < args.steps else np.nan)

    l_offm_end = L["offm"][-1]
    n_offm_max = float(np.nanmax(N["offm"][1:]))
    l_cos_end = L["cos0"][-1]
    n_cos_min = float(np.nanmin(N["cos0"][1:]))
    l_norm_end = float(L["norm"][-1]); n_norm_end = float(np.nanmean(N["norm"]))
    bounded = l_norm_end < 3 * max(1.0, n_norm_end)
    band = l_offm_end > 1.5 * n_offm_max
    decorr = l_cos_end < (n_cos_min - 0.3)
    drift = band or decorr
    print(f"  [detail] natural offm band max={n_offm_max:.3f}, cos floor={n_cos_min:.3f}; "
          f"latent end offm={l_offm_end:.3f}, cos={l_cos_end:.3f}, norm={l_norm_end:.3f}")
    print("\nVERDICT:", ("latent-loop stays within the natural on-manifold band "
                         "=> looping is stable"
                         if not drift else
                         "latent-loop DRIFTS OFF the real-residual manifold "
                         "(bounded norm, but off-manifold distance grows and the "
                         "state decorrelates from its seed) => a readout trained on "
                         "pass-0 residuals will not transfer; loop needs the "
                         "readout (or the model) retrained, as in COCONUT"))


if __name__ == "__main__":
    main()