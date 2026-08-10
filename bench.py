"""bench.py -- fair one-pass readout vs autoregressive loop on closed-set tasks.

Public-benchmark version of the matched-supervision discipline (matched.py):
a ONE-PASS readout of the frozen final-token residual is compared against the
autoregressive loop on a *closed-set* task (BoolQ, RuleTaker, ARC, …). Tasks
live in ``tasks/`` and register via ``tasks.get_task``. The loop baseline is
made FAIR -- full context, a balanced few-shot, scored by next-token log-prob
over the closed answer set (Yes/No or A/B/C/D — a real classifier read, not a
crippled greedy decode).

Readouts (all one pass over passage+question):
  last.linear   logistic probe on the final-token vector
  last.mlp      MLP on the final-token vector (seed sweep, mean +/- std)
  ctx.linear    logistic probe on the mean-pooled FULL-context vector
  ctx.mlp       MLP on the mean-pooled full-context vector
  last.randproj random-Gaussian-projected linear selectivity (deconfound)
  last.mlp.shufl label-shuffle null for the MLP (deconfound)

Loop baselines (next-token log-prob over the answer word):
  loop.zero     zero-shot  (fair conditioning baseline)
  loop.k        few-shot with k balanced exemplars

Also: readout budget curve (n = 64 / 256 / full train) and per-stratum
accuracy (by passage-length tercile, by label, plus task-specific axes like
RuleTaker depth). Run, e.g.:
  python bench.py --local --task boolq --size 0.6B
  python bench.py --local --task ruletaker --size 0.6B --max-train 2000
  python bench.py --local --task arc --size 0.6B
  modal run --detach bench.py --task arc --model 0.6B --gpu t4
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if os.path.exists("/root/bench") and "/root/bench" not in sys.path:
    # bare remote clone: repo mounted to /root/bench/, entrypoint at /root/bench.py
    sys.path.insert(0, "/root/bench")

from common import load_model, resolve_model
from common import n_layers as discover_layers
from probe_main import mlp_probe

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger("bench")

def log(*a):
    LOG.info(" ".join(str(x) for x in a))

PAD_MAX = 384
# Loop arm's own truncation budget. Default 2048 (not PAD_MAX): at 384
# left-truncation eats the prepended few-shot exemplars on long rows and k=8
# silently collapses to zero-shot (truncation caveat). Deliberately pro-baseline
# (loop sees ~5x the readout's evidence) -> readout wins are conservative.
LOOP_PAD_MAX = 2048

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("BENCH_CACHE_DIR",
                          os.path.join(_REPO_ROOT, "cloud_bench_cache"))
# Curated thesis shelf (git-tracked). Distinct from CACHE_DIR, which mirrors the
# Modal volume and is gitignored. Promote winners with --promote after a run.
RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR",
                             os.path.join(_REPO_ROOT, "results"))

# Artifact layout (local CACHE_DIR and Modal /data share the same shape):
#
#   {artifact_root}/{model_slug}/{task}/
#     runs/{run_id}.json     # canonical full metrics+meta (never overwritten)
#     latest.json            # last successful full result for this task
#     stages.jsonl           # progressive stage dumps for THIS task only
#     heads/head_l{L}.npz     # trained readout for this task (latest)
#     cache/vec_t{N}_v{M}_l{L}_p{P}.npz
#     cache/loop_v{M}_l{L}_p{P}.npz
#
# model_slug always comes from resolve_model(alias_or_id) so `0.6B` and
# `Qwen/Qwen3-0.6B` land in the same tree (Qwen__Qwen3-0.6B).
#
# Optional promote: copy the *canonical run file* into RESULTS_DIR under a
# stable name (falls back to latest.json only with a warning), e.g.
#   modal run --detach bench.py ... --promote ruletaker_dsv4_n2k.json
#   modal run --detach bench.py ... --promote results/ruletaker_dsv4_n2k.json
# Prefer runs/{run_id}.json over latest.json — concurrent/smoke runs race
# the latest pointer.


def model_slug(size: str) -> str:
    """Filesystem-safe canonical model id (alias or HF repo → one slug)."""
    return resolve_model(size).replace("/", "__")


def task_dirs(artifact_root: str, size: str, task: str) -> dict[str, str]:
    """Build and create the per-(model, task) directory tree. See CACHE_DIR note."""
    root = os.path.join(artifact_root, model_slug(size), task)
    paths = {
        "task": root,
        "runs": os.path.join(root, "runs"),
        "heads": os.path.join(root, "heads"),
        "cache": os.path.join(root, "cache"),
        "latest": os.path.join(root, "latest.json"),
        "stages": os.path.join(root, "stages.jsonl"),
    }
    for key in ("task", "runs", "heads", "cache"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


# ---------------------------------------------------------------- modal -----
# This file doubles as the Modal runner.
#
# ALWAYS launch cloud runs with --detach:
#   modal run --detach bench.py --model 0.6B --gpu b300 --task boolq ...
# Without --detach, when the local entrypoint exits the ephemeral app is torn
# down and the spawned job dies even though we use .spawn() (seen as "app
# stopped / 0 tasks" with no real work).
#
# The local entrypoint waits for the FunctionCall, then pulls run/latest/heads
# (and optionally cache/) from the bench-results volume into CACHE_DIR so the
# laptop mirrors {model_slug}/{task}/. Use --no-fetch to skip the pull, or
# --fetch-cache to also download vector/loop npz files. Use --promote NAME.json
# to also copy latest.json into results/ (git-tracked thesis shelf).
# Local: `python bench.py --local ...`. Weights: model-weights volume or HF hub.
import modal
import threading

HERE = Path(__file__).resolve().parent
# Top-level files to ship into the Modal image, plus the whole tasks/ package.
_KEEP_ROOT = frozenset({"bench.py", "common.py", "probe_main.py"})
BENCH_VOLUME_NAME = "bench-results"


def _modal_ignore(p) -> bool:
    """Return True to exclude path from the Modal image mount."""
    parts = Path(p).parts
    if not parts:
        return True
    if parts[0] in _KEEP_ROOT and len(parts) == 1:
        return False
    if parts[0] == "tasks":
        if "__pycache__" in parts or any(x.endswith(".pyc") for x in parts):
            return True
        return False
    return True


image = (
    # CUDA 13 (cu130) toolkit base so any CUDA-dependent build/jit succeeds.
    # DeepGEMM's JIT still hangs regardless, so disable it and let the
    # pre-baked Triton FineGrainedFP8 kernel handle FP8 (no JIT compile).
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu22.04",
                              add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda",
          "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR": "1",
          "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .pip_install("torch", "transformers>=4.46", "datasets", "scikit-learn",
                 "numpy", "sentencepiece", "protobuf", "accelerate",
                 "kernels==0.15.2")
    .run_commands(
        # Pre-download the FineGrainedFP8 Triton kernel at BUILD time (into the
        # HF hub cache inside the image layer) so the container never does a
        # slow per-run snapshot_download for the fallback path.
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('kernels-community/finegrained-fp8', "
        "repo_type='kernel')\"",
    )
    .add_local_dir(HERE, "/root/bench", ignore=_modal_ignore)
)

modal_app = modal.App("llm-as-latent-only-bench")

VOLUME_MOUNT = "/data"
bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)

WEIGHTS_MOUNT = "/weights"
weights_volume = modal.Volume.from_name("model-weights", create_if_missing=True)


def volume_rel(path: str) -> str:
    """Map a container abs path under /data to a volume-relative path."""
    if not path:
        return ""
    if path.startswith(VOLUME_MOUNT + "/"):
        return path[len(VOLUME_MOUNT) + 1:]
    if path.startswith(VOLUME_MOUNT):
        return path[len(VOLUME_MOUNT):].lstrip("/")
    return path.lstrip("/")


def _vol_write_file(vol: modal.Volume, remote_path: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
    with open(local_path, "wb") as out:
        for chunk in vol.read_file(remote_path):
            out.write(chunk)


def fetch_modal_artifacts(
    info: dict,
    *,
    local_root: str = CACHE_DIR,
    fetch_cache: bool = False,
) -> list[str]:
    """Pull post-run artifacts from the bench-results volume into local_root.

    Mirrors ``{model_slug}/{task}/`` under ``local_root`` (same tree as local
    runs). Always pulls ``runs/``, ``latest.json``, ``stages.jsonl``, ``heads/``.
    ``cache/`` (vector/loop npz) only if ``fetch_cache`` is True — those can be
    large on full-benchmark runs.
    """
    from modal.types import FileEntryType

    task_dir = (info or {}).get("task_dir")
    if not task_dir:
        print("[fetch] no task_dir in remote result; skip pull")
        return []

    remote_task = volume_rel(task_dir)
    local_task = os.path.join(local_root, remote_task)
    os.makedirs(local_task, exist_ok=True)
    vol = modal.Volume.from_name(BENCH_VOLUME_NAME)

    def _is_file(entry) -> bool:
        return entry.type == FileEntryType.FILE

    def _pull_path(remote: str) -> list[str]:
        got = []
        try:
            entries = vol.listdir(remote, recursive=True)
        except Exception as e:
            # Single file (latest.json / stages.jsonl)
            try:
                local = os.path.join(local_root, remote)
                _vol_write_file(vol, remote, local)
                print(f"[fetch] {remote} -> {local}")
                return [local]
            except Exception as e2:
                print(f"[fetch] skip {remote}: {e2} (listdir: {e})")
                return []
        for e in entries:
            if not _is_file(e):
                continue
            if not fetch_cache and "cache" in e.path.split("/"):
                continue
            local = os.path.join(local_root, e.path)
            try:
                _vol_write_file(vol, e.path, local)
                print(f"[fetch] {e.path} ({e.size} B)")
                got.append(local)
            except Exception as ex:
                print(f"[fetch] failed {e.path}: {ex}")
        return got

    pulled: list[str] = []
    if fetch_cache:
        print(f"[fetch] pulling full task tree {remote_task} -> {local_task}")
        pulled.extend(_pull_path(remote_task))
    else:
        for name in ("latest.json", "stages.jsonl", "heads", "runs"):
            pulled.extend(_pull_path(f"{remote_task}/{name}"))

    print(f"[fetch] pulled {len(pulled)} file(s) under {local_task}")
    return pulled


def resolve_promote_path(dest: str) -> str:
    """Map a promote destination to an absolute path under results/ by default.

    Accepts:
      ruletaker_dsv4_n2k.json          -> {RESULTS_DIR}/ruletaker_dsv4_n2k.json
      results/ruletaker_dsv4_n2k.json  -> {_REPO_ROOT}/results/...
      /abs/path/foo.json               -> unchanged
    """
    dest = (dest or "").strip()
    if not dest:
        raise ValueError("promote path is empty")
    if os.path.isabs(dest):
        return dest
    # Normalize "results/..." relative to repo root; bare names go under RESULTS_DIR.
    norm = dest.replace("\\", "/")
    if norm == "results" or norm.startswith("results/"):
        return os.path.join(_REPO_ROOT, *norm.split("/"))
    return os.path.join(RESULTS_DIR, dest)


def promote_run_json(src: str, dest: str) -> str:
    """Copy a full run JSON (runs/{id}.json preferred) into the results shelf.

    Returns the absolute destination path written. Refuses empty/tiny files so
    a raced empty latest.json cannot wipe a shelf entry.
    """
    if not src or not os.path.isfile(src):
        raise FileNotFoundError(f"promote source missing: {src}")
    if os.path.getsize(src) < 50:
        raise ValueError(
            f"promote source looks empty ({os.path.getsize(src)} B): {src}"
        )
    out = resolve_promote_path(dest)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    shutil.copy2(src, out)
    print(f"[promote] {src} -> {out}")
    return out


def resolve_promote_src(result: dict | None, local_task: str) -> str:
    """Pick the JSON to shelf: canonical run file first, then latest.json.

    ``result`` is the thin Modal return payload (has ``run_file`` / ``run_id``)
    or a full local ``run_bench`` dict (``meta.run_file``).
    """
    result = result or {}
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    run_file = result.get("run_file") or meta.get("run_file")
    run_id = result.get("run_id") or meta.get("run_id")

    candidates: list[tuple[str, str]] = []  # (path, label)
    if run_file:
        # Remote abs path /data/.../runs/id.json → local mirror under local_task
        base = os.path.basename(str(run_file))
        candidates.append(
            (os.path.join(local_task, "runs", base), "run_file(basename)")
        )
        # Also try the path as-is if it already lives under the local tree
        if os.path.isfile(str(run_file)):
            candidates.append((str(run_file), "run_file(local)"))
    if run_id:
        candidates.append(
            (os.path.join(local_task, "runs", f"{run_id}.json"), "run_id")
        )
    candidates.append((os.path.join(local_task, "latest.json"), "latest.json"))

    for path, label in candidates:
        if os.path.isfile(path) and os.path.getsize(path) >= 50:
            if label == "latest.json":
                print(
                    f"[promote] WARN using latest.json (prefer runs/{{run_id}}.json); "
                    f"path={path}"
                )
            else:
                print(f"[promote] source={label} path={path}")
            return path
    tried = ", ".join(p for p, _ in candidates)
    raise FileNotFoundError(
        f"no usable promote source under {local_task}; tried: {tried}"
    )


def gpu_for(model: str) -> str:
    """Pick a GPU class by parameter count (rough). Overridable via --gpu.

    Parses size tokens like ``0.6B``, ``8B``, ``Qwen/Qwen3-8B`` by stripping a
    trailing ``B``/``b`` before ``float`` — bare ``float("0.6B")`` raises and
    used to be swallowed into a silent A100 fallback.
    """
    # DeepSeek-V4 is FP8-quantized: fp8 native needs compute capability >= 8.9
    # (H100/B200/B300). 167GB fits on ONE B300 (288GB) with headroom for the
    # second-pass loop -> no tensor parallelism.
    if "V4" in model:
        return "b300"
    # Last path segment, then last hyphen piece: "Qwen/Qwen3-0.6B" -> "0.6B"
    token = model.rsplit("/", 1)[-1].rsplit("-", 1)[-1].strip()
    if token.lower().endswith("b") and len(token) > 1:
        token = token[:-1]
    try:
        params_b = float(token)  # billions of params
    except (ValueError, TypeError):
        # Unknown id (no *B size token) — conservative default, never silent B300
        return "a100-80gb"
    if params_b >= 60:
        return "a100-80gb"
    if params_b >= 20:
        return "a40"
    return "t4"


@modal_app.function(
    image=image,
    volumes={VOLUME_MOUNT: bench_volume, WEIGHTS_MOUNT: weights_volume},
    timeout=6 * 60 * 60,
    # GPU is set ONLY via .with_options(gpu=...) in main(). Do not put a
    # default here — a hardcoded gpu="b300" made every run look/schedule as
    # B300 even when gpu_for() chose T4 (decorator base vs dynamic pool).
    cpu=32.0,
    memory=32768,
)
def _run_bench_remote(
    model: str,
    max_train: int | None,
    max_val: int | None,
    loop_val: int | None,
    batch: int,
    k_shots: str,
    loop_pad_max: int | None = None,
    run_id: str | None = None,
    task: str = "boolq",
    layer_sweep: str | None = None,
):
    os.environ["HF_HOME"] = os.path.join(VOLUME_MOUNT, "hf_home")
    # Prove which SKU we actually got (dashboard can show the base function).
    try:
        import subprocess
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True).strip()
        print(f"[bench] nvidia-smi: {smi}", flush=True)
    except Exception as e:
        print(f"[bench] nvidia-smi failed: {e}", flush=True)
    # Canonical slug so alias and full HF id share one tree; weights same rule.
    slug = model_slug(model)
    weights_dir = os.path.join(WEIGHTS_MOUNT, slug)
    res = run_bench(
        size=model, device="cuda",
        max_train=max_train, max_val=max_val, loop_val=loop_val,
        batch=batch, k_shots=k_shots, loop_pad_max=loop_pad_max,
        artifact_root=VOLUME_MOUNT, weights_dir=weights_dir,
        run_id=run_id, task=task, layer_sweep=layer_sweep,
    )
    meta = (res or {}).get("meta") or {}
    # Thin payload only: full metrics already live on the volume.
    return {
        "task_dir": meta.get("task_dir"),
        "run_file": meta.get("run_file"),
        "latest_file": meta.get("latest_file"),
        "run_id": meta.get("run_id"),
        "task": meta.get("task"),
        "model": meta.get("model"),
        "model_id": meta.get("model_id"),
        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "finished_at": meta.get("finished_at"),
    }


def cache_path(cache_dir, max_train, max_val, tag="", layer=None, pad_max=PAD_MAX):
    """Path under task cache/. Task + model live in the directory tree, not the
    filename. Layer + pad_max MUST stay in the key (stale-vector FIXME)."""
    ltag = f"_l{layer}" if layer is not None else ""
    if tag == "_loop" or (tag and "loop" in tag):
        fname = f"loop_v{max_val}{ltag}_p{pad_max}.npz"
    else:
        fname = f"vec_t{max_train}_v{max_val}{ltag}_p{pad_max}{tag}.npz"
    path = os.path.join(cache_dir, fname)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def cache_vectors(max_train, max_val, tag="", cache_dir=CACHE_DIR,
                  layer=None, pad_max=PAD_MAX, **arrays):
    """Persist computed hidden vectors + labels so reruns/seed sweeps skip the
    expensive forward passes. Save async stays out of the training loop; a
    blocking save at exit is acceptable here (one-shot per run)."""
    path = cache_path(cache_dir, max_train, max_val, tag,
                      layer=layer, pad_max=pad_max)
    np.savez(path, **arrays)
    log(f"[cache] saved {path}")


def load_cached_vectors(max_train, max_val, keys, tag="", cache_dir=CACHE_DIR,
                        layer=None, pad_max=PAD_MAX):
    path = cache_path(cache_dir, max_train, max_val, tag,
                      layer=layer, pad_max=pad_max)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if not all(k in z for k in keys):
            return None
        return {k: z[k] for k in keys}


# ------------------------------------------------- progressive results save --
def save_stage_results(dirs: dict, stage, data, *, final=False):
    """Append one completed stage to stages.jsonl; on final write latest.json.

    Stages are scoped to one (model, task) tree — no cross-task interleaving.
    """
    jl, js = dirs["stages"], dirs["latest"]
    if final:
        with open(js, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        log(f"[artifacts] wrote {js} (latest)")
        return
    with open(jl, "a") as fh:
        fh.write(json.dumps({"__stage__": stage, **data}, default=str) + "\n")
    log(f"[artifacts] appended stage {stage} -> {jl}")


# ----------------------------------------------------- trained head artifact --
def save_heads(heads_dir, layer, artifacts):
    """Persist the trained readout heads (the 'artifact' a reviewer can load).
    Everything needed to reproduce predictions from a frozen final-token
    residual vector x [d_model]:

      linear: z = (x - mean) / scale;  p = sigmoid(z @ coef + intercept)
      mlp:    z = (x - mu) / sd;  p = sigmoid(w2 @ relu(w1 @ z + b1) + b2)

    Saved as heads/head_l{layer}.npz under the task tree (~2.5 MB).
    Returns path relative to the task dir (for meta.head_file).
    """
    fname = f"head_l{layer}.npz"
    path = os.path.join(heads_dir, fname)
    lin, mlp = artifacts["linear"], artifacts["mlp"]
    clf, sc = lin["clf"], lin["scaler"]
    net = mlp["net"]
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    np.savez(path,
             lin_coef=clf.coef_, lin_intercept=clf.intercept_,
             lin_scaler_mean=sc.mean_, lin_scaler_scale=sc.scale_,
             lin_C=np.array(lin["C"]),
             mlp_w1=sd["0.weight"], mlp_b1=sd["0.bias"],
             mlp_w2=sd["2.weight"], mlp_b2=sd["2.bias"],
             mlp_mu=net._probe_mu, mlp_sd=net._probe_sd,
             mlp_seed=np.array(mlp["seed"]))
    rel = os.path.join("heads", fname)
    log(f"[heads] saved trained readout heads -> {path}")
    return rel


# ---------------------------------------------------------------- readouts --
def to_vecs(model, tok, texts, layer, batch=24, label=""):
    """Final-layer hidden vectors (last-real-token and mean-pooled full context)
    of each passage+question. Captured via a forward hook on ONLY the target
    layer (matched.py pattern) -> holds one layer, not all 37; 4B-safe.
    Returns (X_last, X_mean) as float numpy [n, d_model]."""
    dev = model.device
    last_out, mean_out = [], []
    total = len(texts)
    n_batches = (total + batch - 1) // batch

    def forward_batch(sub):
        enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                  max_length=PAD_MAX).to(dev)
        buf = []

        def hook(m, i, o):
            if isinstance(o, tuple):
                o = o[0]
            # DSV4 layers emit (B, S, G, d) (grouped heads); Qwen emits (B, S, d).
            # Flatten trailing dims -> (B, S, -1) so both read like one token vec.
            o = o.flatten(2)
            buf.append(o.detach().float().cpu())

        h = model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            model(**enc)
        h.remove()
        hs = buf[0]
        # Last-real-token index and context mask off the ATTENTION MASK, not
        # `ids != pad`: valid for left OR right padding (real tokens are
        # contiguous in both; `ids != pad` silently breaks on left-pad).
        am = enc["attention_mask"].cpu()
        lens = am.long().sum(1) - 1
        last_out.append(hs[torch.arange(hs.size(0)), lens])
        mean_out.append((hs * am.float().unsqueeze(-1)).sum(1)
                        / am.float().sum(1).unsqueeze(-1))

    for i, done in enumerate(range(0, total, batch)):
        forward_batch(texts[done:done + batch])
        if (i + 1) % 20 == 0 or done + batch >= total:
            tag = "to_vecs" if not label else label
            log(f"  [{tag}] {min(done + batch, total)}/{total} batches")
    return torch.cat(last_out, 0).numpy(), torch.cat(mean_out, 0).numpy()


def _randproj_acc(Xtr, ytr, Xva, yva, d, s):
    rng = np.random.default_rng(s)
    P = rng.standard_normal((Xtr.shape[1], d)) / (Xtr.shape[1] ** 0.5)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(Xtr @ P, ytr)
    return accuracy_score(yva, clf.predict(Xva @ P))


def randproj_selectivity(Xtr, ytr, Xva, yva, dims=(64, 256), seeds=(0, 1, 2)):
    """Linear probe on random-Gaussian projections. Does the head need the real
    geometry, or would it fire on any (random) projection? Key question for the
    thesis: is the answer-signal localized/structured, or ~isotropic/diffuse?

    Returns {max: best-over-dims random-projection accuracy, d: that dim,
    perm: accuracy when the SAME pipeline runs on a feature-column permutation
    (same linear info, coordinate structure destroyed), noise: same pipeline on
    IID Gaussian features (documents the ~majority-class floor)}."""
    real = {}
    for d in dims:
        real[d] = float(np.mean([_randproj_acc(Xtr, ytr, Xva, yva, d, s)
                                 for s in seeds]))
    d = max(real, key=real.get)

    rng = np.random.default_rng(42)
    perm = rng.permutation(Xtr.shape[1])
    a_p = np.mean([_randproj_acc(Xtr[:, perm], ytr, Xva[:, perm], yva, d, s)
                   for s in seeds])

    rng = np.random.default_rng(7)
    n = Xtr.shape[1]
    G = rng.standard_normal((Xtr.shape[0] + Xva.shape[0], n))
    a_c = np.mean([_randproj_acc(G[:Xtr.shape[0]], ytr,
                                 G[Xtr.shape[0]:], yva, d, s) for s in seeds])

    return {"max": float(max(real.values())), "d": d,
            "perm": float(a_p), "noise": float(a_c)}


def _linear_acc(Xtr, ytr, Xva, yva, sc, C=1.0):
    clf = LogisticRegression(C=C, max_iter=20000, solver="lbfgs")
    clf.fit(sc.transform(Xtr), ytr)
    return accuracy_score(yva, clf.predict(sc.transform(Xva)))


def readout_report(Xtr, ytr, Xva, yva, label, rng, device="cpu",
                   return_artifacts=False, n_classes=None):
    res = {}
    t0 = time.time()
    n_classes = _n_classes_of(ytr, n_classes)
    sc = StandardScaler().fit(Xtr)
    # randproj reports best-over-config (max over dims x seeds); to compare on
    # equal terms the linear readout must also get a config sweep, not a single
    # un-swept C=1.0 fit (bench.py:254 FIXME). Keep the default-C number for
    # continuity and add the swept one.
    # multi_class='auto' handles binary + multinomial for ARC A–D.
    lin = LogisticRegression(
        C=1.0, max_iter=20000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
    log(f"    {label}.linear fit: {time.time() - t0:.0f}s")
    res["last.linear"] = accuracy_score(yva, lin.predict(sc.transform(Xva)))
    best = res["last.linear"]
    best_clf, best_C = lin, 1.0
    for C in (0.01, 0.1, 0.5, 2.0, 10.0):
        clf = LogisticRegression(
            C=C, max_iter=20000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
        a = accuracy_score(yva, clf.predict(sc.transform(Xva)))
        if a > best:
            best, best_clf, best_C = a, clf, C
    res["last.linear.max"] = float(best)
    mlp_stats, best_net, best_seed = mlp_sweep_(
        Xtr, ytr, Xva, yva, device=device, return_best=True,
        n_classes=n_classes)
    res["last.mlp"] = mlp_stats
    res["last.mlp.shufl"] = mlp_sweep_(
        Xtr, rng.permutation(ytr), Xva, yva, device=device,
        n_classes=n_classes)
    rp = randproj_selectivity(Xtr, ytr, Xva, yva)
    res["last.randproj"] = rp
    log(f"    {label}.randproj max={rp['max']:.3f} (d={rp['d']}) "
        f"perm={rp['perm']:.3f} noise={rp['noise']:.3f}")
    if return_artifacts:
        artifacts = {
            "linear": {"clf": best_clf, "scaler": sc, "C": best_C},
            "mlp": {"net": best_net, "seed": best_seed},
        }
        return res, artifacts
    return res


def _n_classes_of(y, n_classes=None):
    if n_classes is not None:
        return int(n_classes)
    y = np.asarray(y)
    if y.dtype == bool or set(np.unique(y.astype(float))).issubset({0.0, 1.0}):
        return 2
    return int(np.max(y)) + 1


def mlp_fit_seeds(Xtr, ytr, seeds=(0, 1, 2, 3), device="cpu", n_classes=None):
    """Fit the seed ensemble once on train only (no stratum-specific training).

    Returns a list of trained nets (each carries ``_probe_mu`` / ``_probe_sd``).
    Use ``mlp_eval_nets`` to score the same heads on many val slices without
    re-fitting — strata are eval filters, not new heads.
    """
    t0 = time.time()
    nets = []
    n_classes = _n_classes_of(ytr, n_classes)
    # mlp_probe requires an X_va for its final forward; dummy one-row val is
    # discarded — we only keep the fitted net + train-side mu/sd.
    dummy_x, dummy_y = Xtr[:1], ytr[:1]
    for s in seeds:
        _, net, _, _ = mlp_probe(
            Xtr, ytr, dummy_x, dummy_y, seed=s, device=device, return_net=True,
            n_classes=n_classes)
        nets.append(net)
        log(f"    mlp seed {s} fit: {time.time() - t0:.0f}s")
    return nets


@torch.no_grad()
def mlp_pred_nets(nets, Xva, n_classes=None):
    """Per-row predictions of a pre-fit seed ensemble, majority vote across
    seeds -> np.int array [n] of class indices. Shares its decode logic with
    ``mlp_eval_nets``; used to persist per-row readout preds for paired
    significance tests (McNemar) against the loop."""
    Xva = np.asarray(Xva)
    if len(Xva) == 0:
        return np.zeros(0, dtype=int)
    votes = []
    for net in nets:
        mu, sd = net._probe_mu, net._probe_sd
        n_cls = getattr(net, "_n_classes", 2)
        dev = next(net.parameters()).device
        xv = torch.tensor((Xva - mu) / (sd + 1e-8), dtype=torch.float32,
                          device=dev)
        out = net(xv).detach().cpu().numpy()
        if n_cls > 2:
            pred = out.argmax(axis=-1)
        else:
            logits = out.squeeze(-1) if out.ndim > 1 else out
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            pred = (p > 0.5).astype(int)
        votes.append(np.asarray(pred, dtype=int))
    V = np.stack(votes)  # [n_seeds, n]
    n_cls = _n_classes_of(np.asarray(V).ravel(), n_classes)
    # majority vote over seeds (ties -> lowest class index)
    return np.apply_along_axis(
        lambda col: np.bincount(col, minlength=n_cls).argmax(), 0, V).astype(int)


@torch.no_grad()
def mlp_eval_nets(nets, Xva, yva):
    """Mean ± std accuracy of a pre-fit seed ensemble on one val slice."""
    if len(Xva) == 0:
        return (float("nan"), float("nan"))
    accs = []
    yva = np.asarray(yva)
    for net in nets:
        mu, sd = net._probe_mu, net._probe_sd
        n_cls = getattr(net, "_n_classes", 2)
        dev = next(net.parameters()).device
        xv = torch.tensor((Xva - mu) / (sd + 1e-8), dtype=torch.float32,
                          device=dev)
        out = net(xv).detach().cpu().numpy()
        if n_cls > 2:
            pred = out.argmax(axis=-1)
        else:
            logits = out.squeeze(-1) if out.ndim > 1 else out
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            pred = p > 0.5
        accs.append(accuracy_score(yva, pred))
    return (float(np.mean(accs)), float(np.std(accs)))


def mlp_sweep_(Xtr, ytr, Xva, yva, seeds=(0, 1, 2, 3), device="cpu",
               return_best=False, n_classes=None):
    """Fit on train, score on one val set (budget curve / one-shot readouts).

    For multi-stratum eval with the *same* train set, prefer
    ``mlp_fit_seeds`` + ``mlp_eval_nets`` so the head is not re-fit per bucket.
    """
    t0 = time.time()
    a = []
    best = (-1.0, None, None)  # (acc, net, seed)
    n_classes = _n_classes_of(ytr, n_classes)
    yva = np.asarray(yva)
    for i, s in enumerate(seeds):
        p, net, _, _ = mlp_probe(Xtr, ytr, Xva, yva, seed=s, device=device,
                                 return_net=True, n_classes=n_classes)
        if n_classes > 2:
            pred = p  # already argmax ints from mlp_probe
        else:
            pred = p > 0.5
        acc = accuracy_score(yva, pred)
        if acc > best[0]:
            best = (acc, net, s)
        a.append(acc)
        log(f"    mlp seed {s}: {time.time() - t0:.0f}s")
    out = (float(np.mean(a)), float(np.std(a)))
    if return_best:
        return out, best[1], best[2]
    return out


def fmt(v):
    if isinstance(v, tuple):
        return f"{v[0]:.3f} +/- {v[1]:.3f}"
    return f"{v:.3f}"


# ------------------------------------------------------------- loop baseline --
def fmt_example(r):
    """Prompt stem ending in ``Answer:`` for loop scoring / residual extract.

    Binary tasks (BoolQ / RuleTaker) use Passage+Question. Parametric MC tasks
    (ARC) leave passage empty and put choices inside ``question``.
    """
    passage = (r.get("passage") or "").strip()
    if passage:
        return f"Passage: {passage}\nQuestion: {r['question']}\nAnswer:"
    return f"Question: {r['question']}\nAnswer:"


def row_label(r) -> int:
    """Integer class index for a row (bool → 0/1, multi-class → int answer)."""
    a = r["answer"]
    if isinstance(a, (bool, np.bool_)):
        return int(a)
    return int(a)


def row_answer_word(r, answer_set) -> str:
    """Surface token the fair loop should prefer for this row."""
    if len(answer_set) == 2 and isinstance(r["answer"], (bool, np.bool_)):
        # Convention: True → first of answer_set if ("Yes","No"), else index.
        # BoolQ/RuleTaker use answer_set ("Yes","No") with True → Yes.
        if answer_set == ("Yes", "No"):
            return "Yes" if r["answer"] else "No"
        return answer_set[int(bool(r["answer"]))]
    return answer_set[int(r["answer"])]


def encode_labels(rows) -> np.ndarray:
    return np.array([row_label(r) for r in rows], dtype=np.int64)


def _answer_token_sets(tok, answer_set):
    """Per-class token-id sets for the continuation after ``Answer:``.

    Includes both space-prefixed and bare forms (tokenizer-dependent). Tokens
    that appear in more than one class are dropped so classes stay disjoint.
    """
    base = tok("Answer:")["input_ids"]
    sets = []
    for ans in answer_set:
        s = set()
        for variant in (ans, " " + ans):
            s.update(tok("Answer:" + variant)["input_ids"][len(base):])
        sets.append(s)
    counts = {}
    for s in sets:
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    shared = {t for t, c in counts.items() if c > 1}
    return [list(s - shared) for s in sets]


def _yes_no_ids(tok):
    # Back-compat wrapper: Yes set, No set (binary).
    ys, ns = _answer_token_sets(tok, ("Yes", "No"))
    return set(ys), set(ns)


def _last_real_index(attn):
    """Index of the final non-pad token per row, valid for LEFT or RIGHT
    padding (assumes the real tokens are contiguous, which both produce)."""
    return attn.size(1) - 1 - attn.flip(1).argmax(1)


def loop_scores(model, tok, texts, exemplar_block=None, batch=24,
                pad_max=PAD_MAX, *, label="loop", cold_start=False,
                answer_set=("Yes", "No")):
    """Fair loop read: full context present, score next token at 'Answer:'
    over ``answer_set`` (Yes/No or A/B/C/D). No sampling, no greedy — a real
    classifier-style continuation read. Left-truncates at `pad_max`.

    Batched: one forward per BATCH of rows. Rows are length-sorted so each
    batch pads to roughly its own length; result order still matches `texts`.

    Returns list[int] class indices (0 = first of answer_set, …). For the
    binary Yes/No convention used by BoolQ, index 1 means Yes.
    """
    dev = model.device
    # Binary Yes/No keeps historical index convention: pred 1 = Yes, 0 = No.
    if tuple(answer_set) == ("Yes", "No"):
        class_sets = _answer_token_sets(tok, ("No", "Yes"))  # 0=No, 1=Yes
        score_order = ("No", "Yes")
    else:
        class_sets = _answer_token_sets(tok, answer_set)
        score_order = answer_set
    for i, ids in enumerate(class_sets):
        if not ids:
            log(f"  [loop {label}] WARN empty token set for {score_order[i]!r}")
    total = len(texts)
    full = [(exemplar_block + "\n\n" + t) if exemplar_block else t
            for t in texts]

    # length-sort (longest first, so any OOM shows up on the first batch rather
    # than 90% of the way in) and remember how to invert the permutation
    order = sorted(range(total), key=lambda i: -len(full[i]))
    preds = [0] * total
    done = 0
    if cold_start:
        log(f"  [loop {label}] scoring {total} rows over {list(score_order)}; "
            f"first forward streams weights -> GPU "
            f"(~minutes on a cold start)...")
    else:
        log(f"  [loop {label}] scoring {total} rows over {list(score_order)} "
            f"(weights already on GPU; re-forward only)...")
    for nb, start in enumerate(range(0, total, batch), 1):
        idx = order[start:start + batch]
        enc = tok([full[i] for i in idx], return_tensors="pt",
                  padding=True, truncation=True, max_length=pad_max).to(dev)
        with torch.no_grad():
            logits = model(**enc).logits
        last = _last_real_index(enc["attention_mask"]).to(logits.device)
        lg = logits[torch.arange(logits.size(0), device=logits.device), last]
        lg = lg.float().log_softmax(-1)
        # (batch, n_classes) — sum log-prob over each class's token id set
        scores = []
        for ids in class_sets:
            if ids:
                scores.append(lg[:, ids].sum(1))
            else:
                scores.append(torch.full(
                    (lg.size(0),), -1e9, device=lg.device, dtype=lg.dtype))
        sc = torch.stack(scores, dim=1)  # [B, C]
        pred_batch = sc.argmax(dim=1).tolist()
        for j, i in enumerate(idx):
            preds[i] = int(pred_batch[j])
        done += len(idx)
        if nb % 10 == 0 or done == total:
            log(f"  [loop {label}] {done}/{total} rows")
    return preds


def _balanced_fewshot(train, k, answer_set):
    """Build a k-shot exemplar block, balanced across classes when possible."""
    if k <= 0:
        return None
    n_cls = len(answer_set)
    by_c = {c: [] for c in range(n_cls)}
    for r in train:
        by_c[row_label(r)].append(r)
    # Round-robin pick so every class appears floor(k/n) or ceil(k/n) times.
    picked = []
    caps = {c: 0 for c in range(n_cls)}
    while len(picked) < k:
        progressed = False
        for c in range(n_cls):
            if len(picked) >= k:
                break
            pool = by_c[c]
            if caps[c] < len(pool):
                picked.append(pool[caps[c]])
                caps[c] += 1
                progressed = True
        if not progressed:
            break
    ex = [fmt_example(r) + " " + row_answer_word(r, answer_set) for r in picked]
    return "\n\n".join(ex) if ex else None


def loop_report(model, tok, train, val, k_shots=(0, 8), batch=24,
                pad_max=PAD_MAX, answer_set=("Yes", "No")):
    """Score loop.k on ``val``. Returns ``(overall_acc, preds_by_key)``.

    ``preds_by_key`` maps ``loop.zero`` / ``loop.k`` -> list[int] class indices
    aligned with ``val`` order (needed for free per-depth loop strata).
    """
    answer_set = tuple(answer_set)
    out = {}
    preds_by_key = {}
    y_true = encode_labels(val)
    for i, k in enumerate(k_shots):
        if hasattr(torch, "cuda") and torch.cuda.is_available() and i:
            torch.cuda.empty_cache()
        block = _balanced_fewshot(train, k, answer_set) if k else None
        texts = [fmt_example(r) for r in val]
        label = f"k={k}" if k else "k=0"
        # Only the first k-shot pass pays the cold weight-stream cost; later
        # passes are separate scoring jobs (new prompts), not a second warm-up.
        preds = loop_scores(model, tok, texts, block, batch=batch,
                            pad_max=pad_max, label=label, cold_start=(i == 0),
                            answer_set=answer_set)
        key = f"loop.{k}" if k else "loop.zero"
        out[key] = accuracy_score(y_true, np.asarray(preds, dtype=int))
        preds_by_key[key] = preds
    return out, preds_by_key


def depth_loop_strata(loop_rows, preds_by_key, last_va, yva, mlp_nets,
                      min_n=8):
    """Per-depth loop vs same-rows MLP on the loop eval set (fair comparison).

    Uses whatever rows were scored by the loop (``loop_rows`` == val[:loop_val]).
    ``mlp_nets`` is the pre-fit full-train seed ensemble (eval only — no re-fit).
    Depth buckets come from each row's ``depth`` field (RuleTaker). Rows without
    a depth key are skipped entirely (BoolQ). No extra GPU forwards for the loop.
    """
    if not loop_rows or not preds_by_key or not mlp_nets:
        return None
    if "depth" not in loop_rows[0]:
        return None
    # Local import: depth_buckets lives on the RuleTaker task module.
    from tasks.ruletaker import depth_buckets
    y = encode_labels(loop_rows)
    buckets = depth_buckets(loop_rows)
    out = {}
    log("  --- depth strata (loop vs mlp, same rows as loop eval) ---")
    for bkey, idxs in buckets.items():
        if len(idxs) < min_n:
            log(f"  depth {bkey}: n={len(idxs)} (skip, too few)")
            continue
        sl = np.array(idxs, dtype=int)
        # last_va / yva are aligned with full val; loop_rows is val[:loop_val]
        # so indices into loop_rows are the same absolute val indices.
        mlp_acc = mlp_eval_nets(mlp_nets, last_va[sl], yva[sl])
        entry = {
            "n": int(len(sl)),
            "pct_true": float(y[sl].mean()),
            "mlp": list(mlp_acc),
        }
        for lk, preds in preds_by_key.items():
            p = np.asarray(preds, dtype=int)
            entry[lk] = float(accuracy_score(y[sl], p[sl]))
        out[str(bkey)] = entry
        loop_bits = " ".join(
            f"{lk}={entry[lk]:.3f}" for lk in preds_by_key)
        log(f"  depth {bkey}: n={len(sl)} mlp={fmt(mlp_acc)} {loop_bits} "
            f"[true={y[sl].mean():.3f}]")
    return out or None


def run_bench(size="0.6B", device=None, max_train=None, max_val=None,
              loop_val=None, batch=8, k_shots="0,8",
              artifact_root=None, cache_dir=None,
              weights_dir=None, result_path=None, run_id=None,
              loop_pad_max: int | None = None, task: str = "boolq",
              layer_sweep: str | None = None):
    """One full bench run (the pipeline the CLI drives), also used directly by
    the Modal app so both share identical code. Returns a dict of scalar
    results (JSON-serializable for cloud orchestration).

    ``task``: registered name in ``tasks.TASKS`` (default ``boolq``). Task-specific
    extra strata (e.g. RuleTaker depth) come from ``TaskSpec.strata``.

    Artifacts land under
    ``{artifact_root}/{model_slug}/{task}/`` (see module-level layout note).
    ``cache_dir`` is accepted only as a deprecated alias for ``artifact_root``.
    ``result_path`` if set writes an extra full JSON copy there (optional).
    """
    import numpy as _np
    np = _np
    import datetime as _dt
    import uuid as _uuid
    from tasks import get_task

    if run_id is None:
        run_id = (_dt.datetime.now().strftime("%Y%m%dT%H%M%S")
                  + "_" + _uuid.uuid4().hex[:6])
    started_at = _dt.datetime.now().isoformat(timespec="seconds")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps"
    probe_dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else device
    rng = np.random.default_rng(0)
    task_spec = get_task(task)
    task = task_spec.name
    # Prefer artifact_root; cache_dir kept as alias so old callers don't break.
    root = artifact_root or cache_dir or CACHE_DIR
    dirs = task_dirs(root, size, task)
    cache_dir = dirs["cache"]
    model_id = resolve_model(size)
    slug = model_slug(size)

    model, tok = load_model(size, device=device, weights_dir=weights_dir)
    # Long passages must NOT lose the "Question: ...\nAnswer:" tail to
    # truncation (bench.py:198 FIXME): right-truncation deleted the question
    # on the top passage-length tercile, and the loop arm (which never
    # truncated) then saw different inputs than the readout. Left-truncate so
    # the answer slot survives; both arms now truncate at PAD_MAX identically.
    tok.truncation_side = "left"
    n_layers_count = discover_layers(model)
    log(f"model={size} ({model_id}) device={device} layers={n_layers_count} "
        f"device_actual={model.device} task={task}")
    log(f"  artifacts -> {dirs['task']}")

    train, val, task_meta, max_train, max_val = task_spec.load(
        max_train, max_val, rng)
    if loop_val is None:
        loop_val = max_val
    # The loop arm gets its OWN truncation budget, independent of the readout's
    # PAD_MAX: at 384 left-truncation eats the prepended few-shot exemplars on
    # long rows, so k=8 collapses to zero-shot (truncation caveat). A larger
    # loop_pad_max is pro-baseline bias (the loop sees MORE evidence than the
    # readout) -- conservative for the headline.
    loop_pad_max = loop_pad_max or LOOP_PAD_MAX
    answer_set = tuple(getattr(task_spec, "answer_set", ("Yes", "No")))
    n_classes = len(answer_set)
    ytr = encode_labels(train)
    yva = encode_labels(val)
    # Binary meta: pct_true = fraction of Yes/True. Multi-class: majority rate.
    if n_classes == 2:
        label_rate = float(yva.mean())
        label_rate_name = "pct_true"
    else:
        counts = np.bincount(yva, minlength=n_classes)
        label_rate = float(counts.max() / max(len(yva), 1))
        label_rate_name = "majority"
    log(f"{task} n_train={len(train)} n_val={len(val)} "
        f"n_classes={n_classes} answer_set={list(answer_set)} "
        f"{label_rate_name}={label_rate:.3f} size={size}")
    if task_meta.get("val_summary") and "depth" in task_meta["val_summary"]:
        log(f"  depth buckets (val): {task_meta['val_summary']['depth']}")
    if task_meta.get("val_summary") and "labels" in task_meta["val_summary"]:
        log(f"  label counts (val): {task_meta['val_summary']['labels']}")

    texts_tr = [fmt_example(r) for r in train]
    texts_va = [fmt_example(r) for r in val]

    log("[1/5] forward-pass hidden vectors (whole dataset)...")
    last_layer = n_layers_count - 1
    vec_cache = load_cached_vectors(
        max_train, max_val,
        ["last_tr", "ctx_tr", "last_va", "ctx_va", "ytr", "yva"],
        cache_dir=cache_dir, layer=last_layer)
    if vec_cache is not None:
        log("[cache] loaded hidden vectors")
        last_tr, ctx_tr = vec_cache["last_tr"], vec_cache["ctx_tr"]
        last_va, ctx_va = vec_cache["last_va"], vec_cache["ctx_va"]
    else:
        last_tr, ctx_tr = to_vecs(model, tok, texts_tr, last_layer,
                                  batch, label="train")
        last_va, ctx_va = to_vecs(model, tok, texts_va, last_layer,
                                  batch, label="val")
        cache_vectors(max_train, max_val,
                      last_tr=last_tr, ctx_tr=ctx_tr,
                      last_va=last_va, ctx_va=ctx_va, ytr=ytr, yva=yva,
                      cache_dir=cache_dir, layer=last_layer)

    log("[2/5] one-pass readouts (final-token / full-context)...")
    res, readout_artifacts = readout_report(
        last_tr, ytr, last_va, yva, "last", rng, device=probe_dev,
        return_artifacts=True, n_classes=n_classes)
    res["ctx.linear"] = readout_report(
        ctx_tr, ytr, ctx_va, yva, "ctx", rng, device=probe_dev,
        n_classes=n_classes)["last.linear"]
    res["ctx.mlp"] = mlp_sweep_(
        ctx_tr, ytr, ctx_va, yva, device=probe_dev, n_classes=n_classes)
    for k in res:
        if isinstance(res[k], dict):
            rp = res[k]
            log(f"  {k:<12} max={rp['max']:.3f} (d={rp['d']}) "
                f"perm={rp['perm']:.3f} noise={rp['noise']:.3f}")
        else:
            log(f"  {k:<16} {fmt(res[k])}")
    result = {"meta": {
        "run_id": run_id,
        "started_at": started_at,
        "model": size,
        "model_id": model_id,
        "model_slug": slug,
        "device": device,
        "n_layers": n_layers_count,
        "n_train": len(train),
        "n_val": len(val),
        "loop_val": loop_val,
        "batch": batch,
        "k_shots": k_shots,
        "pct_true": float(yva.mean()) if n_classes == 2 else label_rate,
        "majority": label_rate,
        "n_classes": n_classes,
        "answer_set": list(answer_set),
        "loop_pad_max": loop_pad_max,
        "pad_max": PAD_MAX,
        "artifact_root": root,
        "task_dir": dirs["task"],
        "artifact_layout": "model_slug/task/v1",
        **task_meta,
    }}
    result.update({k: (float(v[0]), float(v[1])) if isinstance(v, tuple)
                   else (v if isinstance(v, dict) else float(v))
                   for k, v in res.items()})
    save_stage_results(dirs, "readouts", result)

    if layer_sweep:
        # Probe-placement sweep: fit the same one-pass readout at several
        # residual-stream layers to test whether the last-layer tap is optimal
        # (mid-layer residuals often hold relational/factual info earlier).
        # Reuses the exact per-layer vec cache the main path uses, so the
        # sweep is cheap beyond the first forward per layer and a later full
        # run at any swept layer hits the cache.
        sweep_layers = [int(x) for x in str(layer_sweep).split(",")]
        sweep_layers = sorted(
            l for l in set(sweep_layers) if 0 <= l < n_layers_count)
        log(f"[2.5] readout placement sweep over layers {sweep_layers}")
        layer_sweep_res = {}
        # Per-row preds of each layer's 4-seed ensemble (majority vote) -> one
        # combined npz, so paired McNemar between taps (e.g. L18 vs L27) runs
        # CPU-only off artifacts (rowpreds_stats.py --pairs). y_true is the
        # full val — the same rows as the loop rowpreds npz.
        sweep_rowpreds = {"y_true": np.asarray(yva, dtype=int)}
        for l in sweep_layers:
            _vc = load_cached_vectors(
                max_train, max_val,
                ["last_tr", "ctx_tr", "last_va", "ctx_va", "ytr", "yva"],
                cache_dir=cache_dir, layer=l)
            if _vc is not None:
                _ltr, _lva = _vc["last_tr"], _vc["last_va"]
                log(f"[sweep] L{l} vectors from cache")
            else:
                _ltr, _ctr = to_vecs(model, tok, texts_tr, l, batch,
                                     label=f"train@L{l}")
                _lva, _cva = to_vecs(model, tok, texts_va, l, batch,
                                     label=f"val@L{l}")
                cache_vectors(max_train, max_val,
                              last_tr=_ltr, ctx_tr=_ctr,
                              last_va=_lva, ctx_va=_cva,
                              ytr=ytr, yva=yva,
                              cache_dir=cache_dir, layer=l)
            _r, _artifacts = readout_report(
                _ltr, ytr, _lva, yva, "last", rng, device=probe_dev,
                return_artifacts=True, n_classes=n_classes)
            # Persist the trained head at THIS layer (heads/head_l{l}.npz,
            # same artifact format as the pipeline's final-layer head) so a
            # winning mid-layer tap is deployable, not just measured.
            _head_rel = save_heads(dirs["heads"], l, _artifacts)
            _nets = mlp_fit_seeds(_ltr, ytr, device=probe_dev,
                                  n_classes=n_classes)
            sweep_rowpreds[f"readout.mlp.L{l}"] = mlp_pred_nets(
                _nets, _lva, n_classes=n_classes)
            layer_sweep_res[str(l)] = {
                "linear": float(_r["last.linear"]),
                "linear.max": float(_r["last.linear.max"]),
                "mlp": [float(_r["last.mlp"][0]), float(_r["last.mlp"][1])],
                "mlp.shufl": [float(_r["last.mlp.shufl"][0]),
                               float(_r["last.mlp.shufl"][1])],
                "head_file": _head_rel,
            }
            log(f"[sweep] L{l:<3} linear={layer_sweep_res[str(l)]['linear']:.4f} "
                f"max={layer_sweep_res[str(l)]['linear.max']:.4f} "
                f"mlp={layer_sweep_res[str(l)]['mlp'][0]:.4f} "
                f"(head -> {_head_rel})")
        cache_vectors(max_train, max_val, tag="_layersweep_rowpreds",
                      cache_dir=cache_dir, pad_max=PAD_MAX, **sweep_rowpreds)
        _rp_path = cache_path(cache_dir, max_train, max_val,
                              "_layersweep_rowpreds", pad_max=PAD_MAX)
        log(f"[preds] saved per-layer sweep preds -> {_rp_path}")
        best_l = max(sweep_layers,
                     key=lambda l: layer_sweep_res[str(l)]["mlp"][0])
        result["layer_sweep"] = layer_sweep_res
        result["meta"]["layer_sweep"] = layer_sweep
        result["meta"]["layer_sweep_best"] = {
            "layer": best_l,
            "mlp": layer_sweep_res[str(best_l)]["mlp"],
            "head_file": layer_sweep_res[str(best_l)]["head_file"],
        }
        result["meta"]["layer_sweep_rowpreds"] = _rp_path
        save_stage_results(dirs, "layersweep", result)

    log("[3/5] budget curve (last.mlp, final-token)...")
    budget = {}
    for n in (64, 256, max_train):
        if n > len(train):
            continue
        idx = np.arange(n)
        budget[n] = mlp_sweep_(
            last_tr[idx], ytr[idx], last_va, yva, device=probe_dev,
            n_classes=n_classes)
        log(f"  n={n:<6} {fmt(budget[n])}")
    result["budget"] = {str(n): tuple(v) for n, v in budget.items()}
    save_stage_results(dirs, "budget", result)

    log("[4/5] per-stratum (final-token MLP)...")
    # One global head (seed ensemble) on full train; strata only change eval.
    log("  fitting last.mlp seed ensemble once on full train "
        "(strata are eval filters, not re-fits)...")
    stratum_nets = mlp_fit_seeds(
        last_tr, ytr, device=probe_dev, n_classes=n_classes)
    qlen = np.array([len(r["question"]) for r in val])
    plen = np.array([len(r.get("passage") or "") for r in val])
    stratum = {}
    for name, z in (("question-len", qlen), ("passage-len", plen)):
        if name == "passage-len" and float(np.max(z) if len(z) else 0) == 0:
            # Parametric tasks (ARC) have empty passages — skip empty axis.
            continue
        o = np.argsort(z)
        third = max(1, len(o) // 3)
        row = []
        for t, sl in enumerate((o[:third], o[third:2 * third], o[2 * third:])):
            if len(sl) == 0:
                continue
            acc = mlp_eval_nets(stratum_nets, last_va[sl], yva[sl])
            row.append({"acc": list(acc), "true": float(yva[sl].mean())})
            log(f"  {name} tercile {t}: n={len(sl)} {fmt(acc)} "
                f"[y_mean={yva[sl].mean():.3f}]")
        if row:
            stratum[name] = row
    strat_label = {}
    for lab in range(n_classes):
        sl = np.where(yva == lab)[0]
        if len(sl) == 0:
            continue
        acc = mlp_eval_nets(stratum_nets, last_va[sl], yva[sl])
        key = answer_set[lab] if lab < len(answer_set) else str(lab)
        strat_label[key] = list(acc)
        log(f"  label {key}: n={len(sl)} {fmt(acc)}")

    # Task-specific strata (e.g. RuleTaker depth) via TaskSpec.strata.
    extra_strata = task_spec.strata(val) or {}
    for axis, buckets in extra_strata.items():
        axis_out = {}
        log(f"  --- {axis} strata ---")
        for bkey, idxs in buckets.items():
            sl = np.array(idxs, dtype=int)
            if len(sl) < 8:
                log(f"  {axis} {bkey}: n={len(sl)} (skip, too few)")
                continue
            acc = mlp_eval_nets(stratum_nets, last_va[sl], yva[sl])
            axis_out[bkey] = {
                "n": int(len(sl)),
                "pct_true": float(yva[sl].mean()),
                "mlp": list(acc),
            }
            log(f"  {axis} {bkey}: n={len(sl)} {fmt(acc)} "
                f"[y_mean={yva[sl].mean():.3f}]")
        result[f"stratum_{axis}"] = axis_out

    result["stratum"] = stratum
    result["stratum_label"] = strat_label
    save_stage_results(dirs, "stratum", result)

    kk = [int(x) for x in k_shots.split(",")]
    log(f"\n=== loop baseline (next-token log-prob, fair conditioning, "
        f"pad_max={loop_pad_max}, answer_set={list(answer_set)}) ===")
    loop_val_rows = val[: loop_val]
    # Same-rows apples-to-apples: score the *same* full-train heads on the
    # loop's exact subset (no re-fit; different eval set only).
    lsel = np.arange(len(loop_val_rows))
    loop_matched_mlp = mlp_eval_nets(stratum_nets, last_va[lsel], yva[lsel])
    result["last.mlp.loop_matched"] = tuple(loop_matched_mlp)
    log(f"  readout last.mlp on loop's {len(lsel)} rows: "
        f"{fmt(loop_matched_mlp)}")
    lk = [f"loop.{k}" if k else "loop.zero" for k in kk]
    loops = {}
    preds_by_key = None
    loop_cache = load_cached_vectors(0, loop_val, lk, tag="_loop",
                                     cache_dir=cache_dir, layer=last_layer,
                                     pad_max=loop_pad_max)
    if loop_cache is not None:
        log("[cache] loaded loop scores (overall only — no per-row preds; "
            "skip depth×loop strata unless you re-run without cache)")
        loops = {k: float(loop_cache[k]) for k in lk}
    else:
        loops, preds_by_key = loop_report(
            model, tok, train, loop_val_rows, kk, batch=batch,
            pad_max=loop_pad_max, answer_set=answer_set)
        cache_vectors(0, loop_val, tag="_loop", cache_dir=cache_dir,
                      layer=last_layer, pad_max=loop_pad_max, **loops)
        # Persist per-row preds (readout + loop + gold) on the shared loop rows
        # so paired significance tests (McNemar) run CPU-only off artifacts.
        # Cache note: the loop acc cache above stores overall acc only, so
        # per-row preds exist only on fresh (non-cache-hit) runs.
        readout_preds = mlp_pred_nets(stratum_nets, last_va[lsel],
                                      n_classes=n_classes)
        row_preds = {"y_true": np.asarray(yva[lsel], dtype=int),
                     "readout.mlp": readout_preds}
        row_preds.update({k: np.asarray(p, dtype=int)
                          for k, p in preds_by_key.items()})
        cache_vectors(0, loop_val, tag="_rowpreds", cache_dir=cache_dir,
                      layer=last_layer, pad_max=loop_pad_max, **row_preds)
        log(f"[preds] saved per-row preds (readout + {list(preds_by_key)}) "
            f"on {len(lsel)} shared rows")
    for k in loops:
        log(f"  {k:<14} {loops[k]:.3f}")

    result.update(loops)
    # Fair per-depth comparison: same loop_val rows for mlp + loop.k (no extra
    # GPU for the loop — we reuse preds_by_key from the forwards above).
    if preds_by_key is not None:
        depth_loop = depth_loop_strata(
            loop_val_rows, preds_by_key, last_va, yva, stratum_nets)
        if depth_loop is not None:
            result["stratum_depth_loop"] = depth_loop
    result["meta"]["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    head_rel = save_heads(dirs["heads"], last_layer, readout_artifacts)
    result["meta"]["head_file"] = head_rel  # relative: heads/head_l{L}.npz
    run_file = os.path.join(dirs["runs"], f"{run_id}.json")
    result["meta"]["run_file"] = run_file
    result["meta"]["run_file_rel"] = os.path.join("runs", f"{run_id}.json")
    result["meta"]["latest_file"] = dirs["latest"]
    save_stage_results(dirs, "loop", result, final=True)
    with open(run_file, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    log(f"[artifacts] wrote run {run_file} (run_id={run_id})")
    if result_path:
        # Optional extra copy (local CLI convenience / ad-hoc publish path).
        os.makedirs(os.path.dirname(os.path.abspath(result_path)) or ".",
                    exist_ok=True)
        with open(result_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        log(f"[artifacts] extra copy -> {result_path}")

    # If the artifact root is a mounted Modal volume, commit it here so the
    # client's post-run listdir sees the complete tree the moment call.get()
    # returns. Without this, volume writes commit asynchronously and the
    # client can fetch a partial listing (stages + vec cache) and then fail
    # promote with FileNotFoundError because runs/{run_id}.json hadn't
    # committed yet (seen on the 4B Arm A run). Local paths skip this —
    # volume.commit() only exists on Modal's Volume objects.
    vol = globals().get("bench_volume")
    if vol is not None and artifact_root == VOLUME_MOUNT:
        try:
            vol.commit()
            log("[artifacts] volume committed")
        except Exception as e:
            log(f"[artifacts] WARN volume commit failed: {e}")

    return result


@modal_app.local_entrypoint()
def main(
    model: str = "deepseek-ai/DeepSeek-V4-Flash-0731",
    gpu: str | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
    loop_val: int | None = None,
    batch: int = 8,
    k_shots: str = "0,8",
    loop_pad_max: int | None = None,
    task: str = "boolq",
    layer_sweep: str | None = None,
    fetch_cache: bool = False,
    no_fetch: bool = False,
    promote: str = "",
):
    """Local Modal driver. ``promote``: optional path/name under results/ to
    copy the finished run's ``runs/{run_id}.json`` after a successful fetch
    (falls back to latest.json with a warning). Empty string = skip promote."""
    choice = (gpu or gpu_for(model)).lower()
    if choice.startswith("a100"):
        gpu_spec = "a100-80gb"
    elif choice in ("a40", "l40s"):
        gpu_spec = choice
    else:
        gpu_spec = choice  # pass through "t4", "t4g", "h100", "any", etc.

    src = f"--gpu {gpu}" if gpu else f"gpu_for({model!r})"
    print(f"[bench] gpu={gpu_spec}  (from {src})")
    # Must use with_options — base Function has no GPU (see decorator). Chain
    # on the same expression so we never accidentally spawn the base handle.
    call = _run_bench_remote.with_options(gpu=gpu_spec).spawn(
        model=model, max_train=max_train, max_val=max_val,
        loop_val=loop_val, batch=batch, k_shots=k_shots,
        loop_pad_max=loop_pad_max, task=task, layer_sweep=layer_sweep,
    )
    # Volume path is known up front (run_id is not). Print a recovery get so
    # if this client is killed the human/agent can still pull artifacts.
    remote_task = f"{model_slug(model)}/{task}"
    local_task = os.path.join(CACHE_DIR, remote_task)
    recover_cmd = (
        f"modal volume get --force {BENCH_VOLUME_NAME} "
        f"{remote_task} {local_task}"
    )
    print(f"[bench] spawned run call={call.object_id} task={task} gpu={gpu_spec}")
    print(f"[bench] dashboard: {call.get_dashboard_url()}")
    print("[bench] NOTE: launch with `modal run --detach` or this spawn dies "
          "when the client exits.")
    print(f"[bench] RECOVER (if client dies; run after remote finishes):")
    print(f"  {recover_cmd}")
    if promote:
        print(f"[bench] will promote run file (fallback latest.json) -> "
              f"{resolve_promote_path(promote)}")
    print("[bench] waiting for remote (streaming logs); will pull artifacts "
          f"into {local_task} on success"
          + (" [+cache]" if fetch_cache else " [runs/heads/latest only]")
          + (" [no-fetch]" if no_fetch else ""))

    def _stream_logs():
        try:
            for entry in call.logs.stream():
                print(entry.message, end="", flush=True)
        except Exception:
            pass

    log_thread = threading.Thread(target=_stream_logs, daemon=True)
    log_thread.start()
    try:
        result = call.get()
    except Exception as e:
        print(f"[bench] remote failed: {e}")
        raise
    finally:
        log_thread.join(timeout=3)

    summary = {k: (result or {}).get(k)
               for k in ("run_id", "task", "model", "model_id",
                         "n_train", "n_val", "task_dir", "run_file")}
    print("[bench] remote finished:", json.dumps(summary, indent=2, default=str))
    print(f"[bench] RECOVER (same tree): {recover_cmd}")

    if no_fetch:
        print("[bench] --no-fetch: leaving artifacts on the volume only")
        print(f"[bench] pull manually: {recover_cmd}")
        if promote:
            print("[bench] --promote ignored with --no-fetch "
                  "(need a local runs/{run_id}.json; fetch first or promote manually)")
    else:
        # Fetch + promote with retry: even with the remote's volume.commit(),
        # listdir can lag the commit, so a first listing may miss the run
        # JSON. Re-list a few times before giving up (fixes the 4B Arm A
        # FileNotFoundError where promote ran against a 2-file partial pull).
        src = None
        for attempt in range(6):
            fetch_modal_artifacts(
                result or {}, local_root=CACHE_DIR, fetch_cache=fetch_cache)
            if promote:
                try:
                    src = resolve_promote_src(result or {}, local_task)
                    break
                except FileNotFoundError:
                    print(f"[bench] promote source not visible yet "
                          f"(attempt {attempt + 1}/6); retrying in 5s")
                    time.sleep(5)
            else:
                break
        if promote:
            if src is None:
                raise FileNotFoundError(
                    f"promote source never appeared under {local_task} "
                    f"after 6 fetch attempts")
            promote_run_json(src, promote)
    print("[bench] done.")


if __name__ == "__main__":
    # Local:  python bench.py --local --task ruletaker --size 0.6B ...
    # Modal:  modal run --detach bench.py --task ruletaker --model 0.6B ...
    #         (--detach required; client waits + pulls artifacts by default)
    #         --fetch-cache / --no-fetch / --promote NAME.json optional
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        from tasks import task_names
        ap = argparse.ArgumentParser(description="Local bench run")
        ap.add_argument("--local", action="store_true")
        ap.add_argument("--size", default="0.6B")
        ap.add_argument("--task", default="boolq", choices=task_names())
        ap.add_argument("--max-train", type=int, default=None)
        ap.add_argument("--max-val", type=int, default=None)
        ap.add_argument("--loop-val", type=int, default=None)
        ap.add_argument("--batch", type=int, default=8)
        ap.add_argument("--k-shots", default="0,8")
        ap.add_argument("--loop-pad-max", type=int, default=None)
        ap.add_argument("--layer-sweep", default=None,
                        help="comma list of layer indices for probe-placement "
                             "sweep (e.g. --layer-sweep 3,9,15,21,27)")
        ap.add_argument("--device", default=None)
        ap.add_argument("--result-path", default=None,
                        help="extra JSON copy (any path; use results/ for shelf)")
        ap.add_argument("--promote", default=None,
                        help="write into results/ (bare name or results/foo.json)")
        args = ap.parse_args()
        result_path = args.result_path
        if args.promote:
            result_path = resolve_promote_path(args.promote)
        out = run_bench(
            size=args.size, device=args.device, task=args.task,
            max_train=args.max_train, max_val=args.max_val,
            loop_val=args.loop_val, batch=args.batch, k_shots=args.k_shots,
            loop_pad_max=args.loop_pad_max, result_path=result_path,
            layer_sweep=args.layer_sweep,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "meta"},
                         indent=2, default=str))
    else:
        main()
