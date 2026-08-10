"""BoolQ headline chart: one-pass readout vs the fair autoregressive loop.

All four runs are full-val (9427 train / 3270 val, loop pad_max=2048),
read straight from the versioned artifacts in results/:

  Qwen3-0.6B  results_20260808T162558_e83621.json
  Qwen3-4B    results_20260808T162241_cb93ed.json
  Qwen3-8B    results_20260808T154125_17f659.json  (Ext 9)
  DeepSeek-V4 results_20260808T141721_f6d7bb.json  (Ext 8)

Saves boolq_results.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS = [  # plot order: ascending scale, then the frontier MoE
    ("Qwen3-0.6B", "results/results_20260808T162558_e83621.json"),
    ("Qwen3-4B",   "results/results_20260808T162241_cb93ed.json"),
    ("Qwen3-8B",   "results/results_20260808T154125_17f659.json"),
    ("DeepSeek-V4", "results/results_20260808T141721_f6d7bb.json"),
]


def from_artifact(path):
    r = json.load(open(path))
    mlp = r["last.mlp"][0] if isinstance(r["last.mlp"], list) else r["last.mlp"]
    return mlp, r["loop.zero"], r["loop.8"]


names, mlp_v, zero_v, k8_v = [], [], [], []
for name, path in ARTIFACTS:
    m, z, k = from_artifact(path)
    names.append(name); mlp_v.append(m); zero_v.append(z); k8_v.append(k)

x = np.arange(len(names))
w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 5.2))


def bar(off, vals, label, color):
    ax.bar(x + off, vals, w, label=label, color=color, edgecolor="white", linewidth=0.5)
    for xi, v in enumerate(vals):
        ax.text(xi + off, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#222")


bar(-w, mlp_v,  "last.mlp (one-pass readout)", "#2c7fb8")
bar(0.0, zero_v, "loop.zero (fair zero-shot)", "#d95f02")
bar(+w, k8_v,   "loop.8 (fair few-shot)",      "#31a354")

ax.axhline(0.62, color="k", ls="--", lw=0.8)
ax.text(len(names) - 0.5, 0.624, "base rate ~0.62", ha="right", fontsize=7.5, color="#444")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("BoolQ validation accuracy")
ax.set_title("One-pass residual readout vs fair autoregressive loop (BoolQ)\n"
             "full-val (9427 train / 3270 val) · loop pad_max 2048 · single forward pass",
             fontsize=10)
ax.legend(fontsize=8.5, loc="lower right")

plt.tight_layout()
out = "boolq_results.png"
plt.savefig(out, dpi=150)
print("saved", out)
