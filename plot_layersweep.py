"""Ext 14 — readout probe-placement sweep (layer sweep), BoolQ Qwen3-0.6B.

One-pass readout accuracy as a function of the residual-stream layer the
verdict head taps. Shows that the final layer (L27, the repo default) is NOT
optimal: the mid-depth tap (~L18) reads out best. Loop arms (pad 8192, cached
from Arm A / Ext 13) drawn as horizontal references for head-to-head context.

Input:  results/boolq_layersweep_06b.json   (run 20260810T101123_e6f0a6,
        full val 9427/3270, k_shots 0/8/16/32/64, loop_pad_max=8192)
Output: layersweep_06b.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = "results/boolq_layersweep_06b.json"
OUT = "layersweep_06b.png"
COL_MLP = "#d62728"
COL_LIN = "#7f7f7f"
COL_LOOP = "#1f77b4"

d = json.load(open(RUN))
ls = d["layer_sweep"]
layers = sorted(ls.keys(), key=int)
L = np.array([int(l) for l in layers])
mlp = np.array([ls[l]["mlp"][0] for l in layers])
mlp_sd = np.array([ls[l]["mlp"][1] for l in layers])
lin = np.array([ls[l]["linear"] for l in layers])

# loop references (cached from Arm A)
loop64 = d["loop.64"]
loop0 = d["loop.zero"]

best_i = int(np.argmax(mlp))
last_i = len(L) - 1

fig, ax = plt.subplots(figsize=(8.2, 5.0))

# mlp curve (primary) + per-seed std band
ax.plot(L, mlp, "-o", color=COL_MLP, lw=2.0, ms=6, zorder=4,
        label="readout mlp (per-seed mean)")
ax.fill_between(L, mlp - mlp_sd, mlp + mlp_sd, color=COL_MLP, alpha=0.15)
# linear probe (secondary)
ax.plot(L, lin, "--s", color=COL_LIN, lw=1.4, ms=5, zorder=3,
        label="readout linear probe")

for x, y in zip(L, mlp):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=COL_MLP)

# loop references
ax.axhline(loop64, color=COL_LOOP, ls="-.", lw=1.4, alpha=0.85,
           label=f"best loop (64-shot) {loop64:.3f}")
ax.axhline(loop0, color=COL_LOOP, ls=":", lw=1.2, alpha=0.6,
           label=f"loop.zero {loop0:.3f}")

# optimum + final-layer markers
ax.axvline(L[best_i], color=COL_MLP, ls=":", lw=1.0, alpha=0.5)
ax.annotate(f"best tap L{L[best_i]} = {mlp[best_i]:.3f}",
            xy=(L[best_i], mlp[best_i]), xytext=(L[best_i] - 4.2, mlp[best_i] + 0.016),
            ha="center", fontsize=9, fontweight="bold", color=COL_MLP,
            arrowprops=dict(arrowstyle="->", color=COL_MLP, lw=0.9))
ax.annotate(f"final layer L{L[last_i]} = {mlp[last_i]:.3f}\n(repo default; "
            f"{mlp[best_i]-mlp[last_i]:+.3f} below best)",
            xy=(L[last_i], mlp[last_i]), xytext=(21.5, 0.686),
            ha="center", fontsize=8.5, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

ax.set_xlabel("residual-stream layer tapped by the verdict head")
ax.set_ylabel("BoolQ accuracy (full val, n=3270)")
ax.set_title("Readout probe placement sweep — Qwen3-0.6B BoolQ\n"
             "mid-depth tap beats the final layer; both beat the 64-shot loop",
             fontsize=11.5)
ax.set_xticks(L)
ax.set_ylim(min(loop0, lin.min()) - 0.02, mlp.max() + 0.035)
ax.grid(True, axis="y", ls=":", alpha=0.5)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
print(f"best tap: L{L[best_i]} mlp={mlp[best_i]:.4f}  "
      f"final L{L[last_i]} mlp={mlp[last_i]:.4f}  "
      f"delta={mlp[best_i]-mlp[last_i]:+.4f}  best-loop={loop64:.4f}")
