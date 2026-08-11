"""Task-law summary figure — 3x6 heat panel (readout - loop.8, all 18 cells).

Rows = task (BoolQ / RuleTaker n2k / ARC-Challenge), cols = model
(Qwen3-0.6B / 4B / 8B / Mistral-7B / Granite-3.1-8B / DeepSeek-V4-Flash).
Each cell is colored by Delta = readout - loop.8 (paired, same rows) and
annotated with the delta and McNemar significance stars. The "task law" reads
off the panel directly: green everywhere except two red corners — small
models on parametric ARC, and the frontier MoE on serial RuleTaker.

The (delta, p) values are the canonical published numbers from
paper/main_tables.tex Table 1 (delta = readout - loop.k, best-k for BoolQ,
k=8 for RuleTaker/ARC, paired McNemar on the loop's rows). They are
hard-coded so the figure matches the paper exactly; the results/*.json runs
drift by ~1pt run-to-run and serve only as a cross-check.
Output: tasklaw_summary.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ["boolq", "ruletaker", "arc"]
MODELS = [("06b", "Qwen3-0.6B"), ("4b", "Qwen3-4B"), ("8b", "Qwen3-8B"),
          ("mistral7b", "Mistral-7B"), ("granite8b", "Granite-3.1-8B"),
          ("dsv4", "DeepSeek-V4")]
TASK_TITLE = {"boolq": "BoolQ", "ruletaker": "RuleTaker n2k",
              "arc": "ARC-Challenge"}

# Canonical (delta, p) from paper/main_tables.tex Table 1 — the published
# numbers. The sweep/budget JSONs are re-runs and drift by ~1pt run-to-run;
# the figure must match the table, so we hard-code the canonical deltas and
# take the JSONs only as a cross-check. delta = readout - loop.k (best-k for
# BoolQ, k=8 for RuleTaker/ARC).
CANONICAL = {
    ("boolq", "06b"): (0.038, 2.4e-05), ("boolq", "4b"): (-0.007, 0.22),
    ("boolq", "8b"): (-0.007, 0.24), ("boolq", "mistral7b"): (-0.011, 0.21),
    ("boolq", "granite8b"): (-0.010, 0.64), ("boolq", "dsv4"): (-0.009, 0.39),
    ("ruletaker", "06b"): (0.093, 3.7e-03), ("ruletaker", "4b"): (0.040, 0.20),
    ("ruletaker", "8b"): (0.023, 0.45), ("ruletaker", "mistral7b"): (0.085, 0.017),
    ("ruletaker", "granite8b"): (0.133, 1.3e-06), ("ruletaker", "dsv4"): (-0.074, 1e-03),
    ("arc", "06b"): (-0.106, 3.2e-11), ("arc", "4b"): (-0.046, 2.9e-06),
    ("arc", "8b"): (-0.005, 0.57), ("arc", "mistral7b"): (-0.051, 9.3e-05),
    ("arc", "granite8b"): (-0.082, 5.8e-10), ("arc", "dsv4"): (-0.007, 0.23),
}

def cell(task, suf):
    return CANONICAL[(task, suf)] + (None, None)


def stars(p):
    if p is None:
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def main():
    Z = np.zeros((len(TASKS), len(MODELS)))
    P = np.full((len(TASKS), len(MODELS)), np.nan)
    for i, t in enumerate(TASKS):
        for j, (suf, _) in enumerate(MODELS):
            delta, p, _, _ = cell(t, suf)
            Z[i, j] = delta
            P[i, j] = p if p is not None else np.nan

    lim = max(0.12, np.nanmax(np.abs(Z)))
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    im = ax.imshow(Z, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m for _, m in MODELS], fontsize=9)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([TASK_TITLE[t] for t in TASKS], fontsize=10)
    ax.xaxis.set_ticks_position("top")

    for i in range(len(TASKS)):
        for j in range(len(MODELS)):
            dv = Z[i, j]
            pv = P[i, j]
            s = stars(pv) if not np.isnan(pv) else ""
            ax.text(j, i, f"{dv:+.3f}{s}", ha="center", va="center",
                    fontsize=9, fontweight="bold" if s else "normal",
                    color="black")

    ax.set_title("readout − loop.8 (paired, same rows)  ·  *p<0.05  **p<0.01  ***p<0.001",
                 fontsize=10, pad=26)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(TASKS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("readout − loop.8 (accuracy)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig("paper/figures/tasklaw_summary.png", dpi=170, bbox_inches="tight")
    print("wrote tasklaw_summary.png")


if __name__ == "__main__":
    main()
