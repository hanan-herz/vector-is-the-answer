"""Confidence smoke test on Qwen3-4B — pure head-forward, no model load.

Smoke check that trained read heads emit *useable* confidence: for each task
head on its cached val residual, report accuracy and calibration diagnostics
(reliability-diagram bins, ECE, Brier, threshold behavior). All from the head's
softmax probs — no re-fitting, no temperature tuning. This confirms the
NOTES.md claim that confidence is a free byproduct of the usual heads.

Runs entirely on cached residuals pulled from the Modal bench-results volume
(results_4b_residuals/*.npz); load_* helpers mirror multihead_route.load_head_npz
so this is standalone (no torch / model required).
"""
from __future__ import annotations

import numpy as np

RESID_DIR = "results_4b_residuals"

# per-task: {name: (head_npz, residual_cache, n_train)}
TASKS = {
    "boolq": dict(
        head="results/head_Qwen_Qwen3-4B_l35.npz",
        res=f"{RESID_DIR}/boolq.npz"),
    "ruletaker": dict(
        head="cloud_bench_cache/Qwen__Qwen3-4B/ruletaker/heads/head_l35.npz",
        res=f"{RESID_DIR}/ruletaker.npz"),
    "arc": dict(
        head="cloud_bench_cache/Qwen__Qwen3-4B/arc/heads/head_l35.npz",
        res=f"{RESID_DIR}/arc.npz"),
}


def load_head_npz(path: str) -> dict:
    z = np.load(path)
    n_out = int(z["mlp_w2"].shape[0])
    return {
        "multi": n_out > 1,
        "n_classes": n_out,
        "mlp_w1": z["mlp_w1"].astype(np.float32),
        "mlp_b1": z["mlp_b1"].astype(np.float32),
        "mlp_w2": z["mlp_w2"].astype(np.float32),
        "mlp_b2": z["mlp_b2"].astype(np.float32),
        "mlp_mu": z["mlp_mu"].astype(np.float32),
        "mlp_sd": z["mlp_sd"].astype(np.float32),
    }


def head_forward(head: dict, X: np.ndarray):
    """Return (pred, conf, probs). Binary: pred {0,1}, conf = max(p,1-p),
    probs[n,2]. Multi: pred argmax, conf = max softmax, probs[n,C]."""
    z = (X - head["mlp_mu"]) / (head["mlp_sd"] + 1e-8)
    h = np.maximum(0.0, z @ head["mlp_w1"].T + head["mlp_b1"])
    logits = h @ head["mlp_w2"].T + head["mlp_b2"]
    if head["multi"]:
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        probs = e / e.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1).astype(np.int64)
        conf = probs.max(axis=1)
        return pred, conf, probs
    logits = logits.reshape(-1)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    probs = np.stack([1.0 - p1, p1], axis=1)
    pred = (p1 > 0.5).astype(np.int64)
    conf = np.maximum(p1, 1.0 - p1)
    return pred, conf, probs


def expected_calibration_error(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, tot = 0.0, len(conf)
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n == 0:
            bin_stats.append(dict(bin=(lo, hi), n=0))
            continue
        acc = correct[m].mean()
        ece += n / tot * abs(acc - conf[m].mean())
        bin_stats.append(dict(bin=(lo, hi), n=n, conf=conf[m].mean(), acc=acc))
    return ece, bin_stats


def smoke(name, cfg):
    head = load_head_npz(cfg["head"])
    z = np.load(cfg["res"])
    X = np.asarray(z["last_va"], dtype=np.float32)
    y = np.asarray(z["yva"]).astype(np.int64)
    pred, conf, _ = head_forward(head, X)
    correct = (pred == y).astype(np.float64)
    acc = correct.mean()
    # Brier = mean((p_pred - 1)^2) over predicted class; = mean((1-conf)^2 for correct)
    # Standard multiclass Brier: mean over all classes of (p_c - y_c)^2.
    # For binary: p1 vs y. We'll use the usual "conf vs correctness":
    brier = ((conf - correct) ** 2).mean()
    ece, bins = expected_calibration_error(conf, correct)

    print(f"\n=== {name}  (n={len(X)}, acc={acc:.4f}) ===")
    print(f"  ECE={ece:.4f}  Brier={brier:.4f}  mean_conf={conf.mean():.4f}")
    print(f"  {'bin':>16} {'n':>6} {'conf':>6} {'acc':>6}")
    for b in bins:
        if b["n"] == 0:
            continue
        nm = f"[{b['bin'][0]:.2f},{b['bin'][1]:.2f})"
        print(f"  {nm:>16} {b['n']:>6} {b['conf']:>6.3f} {b['acc']:>6.3f}")
    # threshold behavior
    print("  thresholds:")
    for th in (0.99, 0.97, 0.95, 0.9, 0.8, 0.7, 0.6):
        m = conf >= th
        if m.sum():
            print(f"   conf>={th:5}: n={m.sum():5}  keep_acc={correct[m].mean():.3f}")
    return dict(name=name, n=len(X), acc=acc, ece=ece, brier=brier,
                mean_conf=float(conf.mean()))


if __name__ == "__main__":
    print(f"Qwen3-4B confidence smoke (pure head-forward over cached residuals)\n")
    out = {}
    for name, cfg in TASKS.items():
        out[name] = smoke(name, cfg)
    print("\n--- summary ---")
    for k, v in out.items():
        print(f"  {k:10s} acc={v['acc']:.4f} ece={v['ece']:.4f} "
              f"brier={v['brier']:.4f} mean_conf={v['mean_conf']:.4f}")