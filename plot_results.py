"""Plot the latent-probe vs decoder-readout results across model sizes.

Saves two full-size figures (same figsize as boolq_results.png):
  probe_results_bars.png  — single-pass readouts vs decoder
  probe_results_layers.png — probe accuracy by residual-stream layer
Also writes a side-by-side composite probe_results.png for README use.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Collected from probe_main.py runs on Qwen3, multi-hop transitivity task.
# per-layer latent probe val accuracy by model size (--per-depth 700).
per_layer = {
    "0.6B": np.array([0.491, 0.811, 0.943, 0.943, 0.943, 0.954, 0.954, 0.960, 0.931]),
    "4B":   np.array([0.491, 0.869, 0.949, 0.943, 0.954, 0.983, 0.971, 0.966, 0.966]),
    "8B":   np.array([0.486, 0.920, 0.966, 0.960, 0.977, 0.977, 0.977, 0.983, 0.971]),
}
layer_names = ["L0", "L3", "L6", "L9", "L12", "L15", "L18", "L21", "L27/35"]
last_probe = {"0.6B": 0.931, "4B": 0.966, "8B": 0.971}
lm_head = {"0.6B": 0.491, "4B": 0.846, "8B": 0.680}
mlp = {"0.6B": 0.909, "4B": 0.954, "8B": 0.966}
zero_shot = {"0.6B": 0.463, "4B": 0.474, "8B": 0.474}
surface = {"0.6B": 0.531, "4B": 0.531, "8B": 0.531}
shuffle_null = {"0.6B": 0.480, "4B": 0.440, "8B": 0.526}
sizes = ["0.6B", "4B", "8B"]
x = np.arange(len(sizes))
w = 0.18

# Match boolq_results.png panel size (plot_boolq.py: figsize=(9.5, 5.2)).
FIGSIZE = (9.5, 5.2)
DPI = 150
OUT_DIRS = [Path("."), Path("paper")]


def _save(fig, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print("saved", path)


def plot_bars(ax) -> None:
    ax.bar(x - 1.5 * w, [last_probe[s] for s in sizes], w,
           label="Latent probe (last layer)", color="#2c7fb8")
    ax.bar(x - 0.5 * w, [mlp[s] for s in sizes], w,
           label="MLP (single pass)", color="#7a5195")
    ax.bar(x + 0.5 * w, [lm_head[s] for s in sizes], w,
           label="Decoder (few-shot)", color="#31a354")
    ax.bar(x + 1.5 * w, [zero_shot[s] for s in sizes], w,
           label="Decoder (zero-shot)", color="#d95f02")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Single-pass readouts vs decoder\n(multi-hop transitivity, Qwen3)")
    ax.legend(fontsize=9, loc="lower right")
    for xi, s in zip(x, sizes):
        ax.text(xi - 1.5 * w, last_probe[s] + 0.02, f"{last_probe[s]:.2f}",
                ha="center", fontsize=9)
        ax.text(xi - 0.5 * w, mlp[s] + 0.02, f"{mlp[s]:.2f}",
                ha="center", fontsize=9)
        ax.text(xi + 0.5 * w, lm_head[s] + 0.02, f"{lm_head[s]:.2f}",
                ha="center", fontsize=9)


def plot_layers(ax) -> None:
    for s in sizes:
        ax.plot(range(len(per_layer[s])), per_layer[s], marker="o", label=s)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Transformer layer (residual stream)")
    ax.set_ylabel("Probe validation accuracy")
    ax.set_title("Latent structure emerges through depth")
    ax.legend(fontsize=9)


# Full-size standalone figures (paper)
fig, ax = plt.subplots(figsize=FIGSIZE)
plot_bars(ax)
fig.tight_layout()
_save(fig, "probe_results_bars.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=FIGSIZE)
plot_layers(ax)
fig.tight_layout()
_save(fig, "probe_results_layers.png")
plt.close(fig)

# Side-by-side composite (README / RESULTS_SYNTHETIC)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
plot_bars(axes[0])
plot_layers(axes[1])
fig.tight_layout()
_save(fig, "probe_results.png")
plt.close(fig)