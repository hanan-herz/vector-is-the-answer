"""Ext 14/14b — readout probe-placement sweeps (layer sweeps).

Two panels: BoolQ Qwen3-0.6B (full val 9427/3270, taps 1/5/9/13/18/23/27) and
RuleTaker n2k Qwen3-4B (2000/1000, taps 1/6/12/18/24/30/35). One-pass readout
accuracy as a function of the residual-stream layer the verdict head taps.
The repo default (final layer) is NOT optimal: mid-depth plateau at both
scales; significant at 4B/RuleTaker (paired McNemar, same rows), ns at
0.6B/BoolQ. Loop arms drawn as horizontal references for head-to-head context.

Inputs:
  results/boolq_layersweep_06b.json{,_paired.json}   (run 20260810T104749_6aea90)
  results/ruletaker_layersweep_4b.json{,_paired.json} (run 20260810T105307_b42924)
Output: layersweep_placement.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL_MLP = "#d62728"
COL_LIN = "#7f7f7f"
COL_LOOP = "#1f77b4"

PANELS = [
    dict(run="results/boolq_layersweep_06b.json",
         paired="results/boolq_layersweep_06b_paired.json",
         title="Qwen3-0.6B · BoolQ (full val, n=3270)",
         loop_refs=[("loop.64", "best loop (64-shot)"), ("loop.zero", "loop.zero")]),
    dict(run="results/ruletaker_layersweep_4b.json",
         paired="results/ruletaker_layersweep_4b_paired.json",
         title="Qwen3-4B · RuleTaker n2k (n=1000)",
         loop_refs=[("loop.8", "loop.8 (400 rows)"), ("loop.zero", "loop.zero (400 rows)")]),
]


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

for ax, cfg in zip(axes, PANELS):
    d = json.load(open(cfg["run"]))
    paired = {r["pair"]: r for r in json.load(open(cfg["paired"]))["reports"]}
    ls = d["layer_sweep"]
    layers = sorted(ls.keys(), key=int)
    L = np.array([int(l) for l in layers])
    mlp = np.array([ls[l]["mlp"][0] for l in layers])
    mlp_sd = np.array([ls[l]["mlp"][1] for l in layers])
    lin = np.array([ls[l]["linear"] for l in layers])

    best_i = int(np.argmax(mlp))
    last_i = len(L) - 1

    ax.plot(L, mlp, "-o", color=COL_MLP, lw=2.0, ms=6, zorder=4,
            label="readout mlp (per-seed mean)")
    ax.fill_between(L, mlp - mlp_sd, mlp + mlp_sd, color=COL_MLP, alpha=0.15)
    ax.plot(L, lin, "--s", color=COL_LIN, lw=1.4, ms=5, zorder=3,
            label="readout linear probe")
    for x, y in zip(L, mlp):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=COL_MLP)

    for j, (key, lab) in enumerate(cfg["loop_refs"]):
        v = d[key]
        ax.axhline(v, color=COL_LOOP, ls="-." if j == 0 else ":", lw=1.4,
                   alpha=0.85 if j == 0 else 0.6, label=f"{lab} {v:.3f}")

    ax.axvline(L[best_i], color=COL_MLP, ls=":", lw=1.0, alpha=0.5)

    # paired McNemar: best tap vs final layer (same rows, seed-vote preds)
    pair_key = f"readout.mlp.L{L[best_i]} vs readout.mlp.L{L[last_i]}"
    r = paired[pair_key]
    p, sym = r["mcnemar_p"], sig_stars(r["mcnemar_p"])
    pstr = f"p={p:.1e}" if p < 0.01 else f"p={p:.2f}"
    ax.annotate(
        f"best tap L{L[best_i]} = {mlp[best_i]:.3f}\n"
        f"vs final L{L[last_i]} = {mlp[last_i]:.3f}\n"
        f"{r['diff']:+.3f}  {pstr} {sym}",
        xy=(L[best_i], mlp[best_i]),
        xytext=(0.52, 0.30), textcoords="axes fraction",
        ha="left", fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel("residual-stream layer tapped by the verdict head")
    ax.set_title(cfg["title"], fontsize=12)
    ax.set_xticks(L)
    lo = min([d[k] for k, _ in cfg["loop_refs"]] + list(lin)) - 0.02
    ax.set_ylim(lo, mlp.max() + 0.04)
    ax.grid(True, axis="y", ls=":", alpha=0.5)

axes[0].set_ylabel("accuracy")
axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
axes[1].legend(fontsize=8.5, loc="lower right", framealpha=0.9)

fig.suptitle("Readout probe placement sweep — mid-depth taps beat the final layer "
             "(significant at 4B/RuleTaker, ns at 0.6B/BoolQ)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("layersweep_placement.png", dpi=150, bbox_inches="tight")
print("saved layersweep_placement.png")
