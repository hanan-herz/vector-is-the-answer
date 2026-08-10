"""Ext 14(a–d) — readout probe-placement sweeps (layer sweeps).

Four-panel grid: BoolQ (0.6B / 8B) and RuleTaker n2k (4B / 8B). One-pass
readout accuracy as a function of the residual-stream layer the verdict head
taps. The repo default (final layer) is NOT optimal — mid-depth taps beat it
at 3/4 sweeps (ns only at 0.6B/BoolQ). Paired-McNemar annotations per panel.
Loop arms are horizontal references (one shared scale per task).

Inputs:
  results/boolq_layersweep_06b.json{,_paired.json}   (run 20260810T104749_6aea90)
  results/boolq_layersweep_8b.json{,_paired.json}    (run 20260810T110623_98703d)
  results/ruletaker_layersweep_4b.json{,_paired.json}  (run 20260810T105307_b42924)
  results/ruletaker_layersweep_8b.json{,_paired.json}  (run 20260810T110623_69b106)
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
    # left column: BoolQ
    dict(run="results/boolq_layersweep_06b.json",
         paired="results/boolq_layersweep_06b_paired.json",
         title="Qwen3-0.6B · BoolQ (n=3270)",
         loop_refs=[("loop.64", "best loop (64-shot)"), ("loop.zero", "loop.zero")]),
    # right column: RuleTaker
    dict(run="results/ruletaker_layersweep_4b.json",
         paired="results/ruletaker_layersweep_4b_paired.json",
         title="Qwen3-4B · RuleTaker n2k (n=1000)",
         loop_refs=[("loop.8", "loop.8 (400 rows)"), ("loop.zero", "loop.zero (400 rows)")]),
    # left column: BoolQ
    dict(run="results/boolq_layersweep_8b.json",
         paired="results/boolq_layersweep_8b_paired.json",
         title="Qwen3-8B · BoolQ (n=3270)",
         loop_refs=[("loop.64", "best loop (64-shot)"), ("loop.zero", "loop.zero")]),
    # right column: RuleTaker
    dict(run="results/ruletaker_layersweep_8b.json",
         paired="results/ruletaker_layersweep_8b_paired.json",
         title="Qwen3-8B · RuleTaker n2k (n=1000)",
         loop_refs=[("loop.8", "loop.8 (400 rows)"), ("loop.zero", "loop.zero (400 rows)")]),
]


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))

for ax, cfg in zip(axes.flat, PANELS):
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
        ax.axhline(v, color=COL_LOOP, ls="--" if j == 0 else ":", lw=1.4,
                   alpha=0.85 if j == 0 else 0.6, label=f"{lab} {v:.3f}")

    ax.axvline(L[best_i], color=COL_MLP, ls=":", lw=1.0, alpha=0.5)

    # paired McNemar: best tap vs final layer (same rows, seed-vote preds)
    pair_key = f"readout.mlp.L{L[best_i]} vs readout.mlp.L{L[last_i]}"
    r = paired[pair_key]
    p, sym = r["mcnemar_p"], sig_stars(r["mcnemar_p"])
    pstr = f"p={p:.1e}" if p < 0.01 else f"p={p:.2f}"

    # annotate the best-point callout: position depends on whether best is
    # early (right-side callout) or late (left-side callout relative to point)
    frac_x, ha = (0.52, "left") if best_i < 3 else (0.40, "right")
    ax.annotate(
        f"best tap L{L[best_i]} = {mlp[best_i]:.3f}\n"
        f"vs final L{L[last_i]} = {mlp[last_i]:.3f}\n"
        f"{r['diff']:+.3f}  {pstr} {sym}",
        xy=(L[best_i], mlp[best_i]),
        xytext=(frac_x, 0.30), textcoords="axes fraction",
        ha=ha, fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel("residual-stream layer tapped by the verdict head")
    ax.set_title(cfg["title"], fontsize=12)
    ax.set_xticks(L)
    lo = min([d[k] for k, _ in cfg["loop_refs"]] + list(lin)) - 0.02
    ax.set_ylim(lo, mlp.max() + 0.04)
    ax.grid(True, axis="y", ls=":", alpha=0.5)

# Shared legends per task column
axes[0, 0].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
axes[0, 1].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
axes[1, 0].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
axes[1, 1].legend(fontsize=8.5, loc="upper left", framealpha=0.9)

# Faint vertical split between BoolQ (left) and RuleTaker (right) columns
fig.canvas.draw()  # resolve positions
if hasattr(fig.canvas, "get_renderer"):
    r = fig.canvas.get_renderer()
else:
    r = fig.canvas.renderer
b0 = axes[0, 0].get_position()   # left-column top
b1 = axes[0, 1].get_position()   # right-column top
b3 = axes[1, 1].get_position()   # right-column bottom
x_mid = (b0.x1 + b1.x0) / 2      # gap centre
line = plt.Line2D([x_mid, x_mid], [b3.y0, b0.y1],
                  transform=fig.transFigure, color="0.5", lw=1.5, ls="--",
                  alpha=0.4, zorder=0)
fig.add_artist(line)

fig.suptitle("Readout probe placement sweep — 3/4 sweeps show a significant mid-depth advantage",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("layersweep_placement.png", dpi=150, bbox_inches="tight")
print("saved layersweep_placement.png")
