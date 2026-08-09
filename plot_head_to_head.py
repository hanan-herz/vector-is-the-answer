"""Head-to-head: BoolQ (full-val) · RuleTaker (n2k) · ARC-Challenge (full).

Same four models, same three bars:
  last.mlp · loop.zero · loop.8

Protocols differ by design —
  BoolQ:      train 9427 / val 3270 / loop_val 3270
  RuleTaker:  train 2000 / val 1000 / loop_val 400  (n2k pilot)
  ARC:        train 1117 / test 1165 / loop_val 1165 (full Challenge, 4-way)

Saves:
  head_to_head_three_tasks.png
  head_to_head_boolq_ruletaker.png  (same figure; legacy name)

Full paths printed on write.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

BOOLQ = [  # ascending scale, then frontier MoE
    ("Qwen3-0.6B", "results/results_20260808T162558_e83621.json"),
    ("Qwen3-4B", "results/results_20260808T162241_cb93ed.json"),
    ("Qwen3-8B", "results/results_20260808T154125_17f659.json"),
    ("DeepSeek-V4", "results/results_20260808T141721_f6d7bb.json"),
]

RULETAKER = [
    ("Qwen3-0.6B", "results/ruletaker_qwen06b_n2k.json"),
    ("Qwen3-4B", "results/ruletaker_qwen4b_n2k.json"),
    ("Qwen3-8B", "results/ruletaker_qwen8b_n2k.json"),
    ("DeepSeek-V4", "results/ruletaker_dsv4_n2k.json"),
]

ARC = [
    ("Qwen3-0.6B", "results/arc_qwen06b.json"),
    ("Qwen3-4B", "results/arc_qwen4b.json"),
    ("Qwen3-8B", "results/arc_qwen8b.json"),
    ("DeepSeek-V4", "results/arc_dsv4.json"),
]

COLORS = {
    "mlp": "#2c7fb8",
    "zero": "#d95f02",
    "k8": "#31a354",
}


def _acc(x):
    return float(x[0]) if isinstance(x, (list, tuple)) else float(x)


def load_triplet(path: Path):
    r = json.loads(path.read_text())
    return (
        _acc(r["last.mlp"]),
        float(r["loop.zero"]),
        float(r["loop.8"]),
        _acc(r["last.mlp.loop_matched"]) if "last.mlp.loop_matched" in r else None,
    )


def panel(ax, artifacts, *, title, ylabel, base_rate=None, note=None,
          ylim=(0.5, 1.0), show_legend=True):
    names, mlp_v, zero_v, k8_v, matched_v = [], [], [], [], []
    for name, rel in artifacts:
        path = ROOT / rel
        m, z, k, matched = load_triplet(path)
        names.append(name)
        mlp_v.append(m)
        zero_v.append(z)
        k8_v.append(k)
        matched_v.append(matched)

    x = np.arange(len(names))
    w = 0.26

    def bar(off, vals, label, color):
        ax.bar(
            x + off, vals, w, label=label, color=color,
            edgecolor="white", linewidth=0.5,
        )
        for xi, v in enumerate(vals):
            ax.text(
                xi + off, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5, color="#222",
            )

    bar(-w, mlp_v, "last.mlp (one-pass)", COLORS["mlp"])
    bar(0.0, zero_v, "loop.zero", COLORS["zero"])
    bar(+w, k8_v, "loop.8 (fair few-shot)", COLORS["k8"])

    # Matched readout marker when available (same n as loop).
    if any(v is not None for v in matched_v):
        for xi, mv in enumerate(matched_v):
            if mv is None:
                continue
            ax.plot(
                xi - w, mv, marker="D", color="#08306b", markersize=4.5,
                zorder=5, linestyle="none",
            )
        ax.plot(
            [], [], marker="D", color="#08306b", markersize=4.5,
            linestyle="none", label="last.mlp.loop_matched",
        )

    if base_rate is not None:
        ax.axhline(base_rate, color="k", ls="--", lw=0.8)
        ax.text(
            len(names) - 0.5, base_rate + 0.008,
            f"chance/base ~{base_rate:.2f}",
            ha="right", fontsize=6.5, color="#444",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5)
    if note:
        ax.text(
            0.5, -0.16, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="#444",
        )
    if show_legend:
        ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=False)

    panel(
        axes[0],
        BOOLQ,
        title="BoolQ — full validation",
        ylabel="Accuracy",
        base_rate=0.62,
        note="train 9427 · val 3270 · loop full · Yes/No",
        ylim=(0.55, 1.0),
        show_legend=True,
    )
    panel(
        axes[1],
        RULETAKER,
        title="RuleTaker — n2k pilot",
        ylabel="Accuracy",
        base_rate=0.50,
        note="train 2000 · val 1000 · loop 400 · in-context rules",
        ylim=(0.50, 0.90),
        show_legend=False,
    )
    panel(
        axes[2],
        ARC,
        title="ARC-Challenge — full test",
        ylabel="Accuracy",
        base_rate=0.25,
        note="train 1117 · test 1165 · loop full · A–D · parametric",
        ylim=(0.20, 1.05),
        show_legend=False,
    )

    fig.suptitle(
        "One-pass residual readout vs fair autoregressive loop\n"
        "BoolQ (passage) · RuleTaker (in-context rules) · ARC (parametric knowledge)",
        fontsize=11.5,
        y=1.03,
    )
    fig.tight_layout()
    outs = [
        ROOT / "head_to_head_three_tasks.png",
        ROOT / "head_to_head_boolq_ruletaker.png",  # legacy alias (now 3 panels)
    ]
    for out in outs:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("saved", out.resolve())


if __name__ == "__main__":
    main()
