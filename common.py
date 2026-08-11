"""Shared model/config helpers for the latent-probe experiments.

Also the single home of the artifact-cache layout (PAD_MAX/CACHE_DIR/
task_dirs/cache_path/cache_vectors/load_cached_vectors): bench.py and the
local analysis scripts (multihead_route / gate_ar / paired_test) all import
from here so cache keys are defined exactly once."""
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# Keep the CLI/cloud logs readable: silence transformers' tqdm ("Loading
# weights: xx%") and let our own step logging own the output. The cleaner the
# cloud/terminal stream, the easier the status reads at a glance.
hf_logging.disable_progress_bar()

ALLOWED_SIZE = {
    "0.6B": 28, "1.7B": 28, "4B": 36, "8B": 36,
}

SIZES = sorted(ALLOWED_SIZE)


def model_name(size):
    return f"Qwen/Qwen3-{size}"


def resolve_model(name):
    """Return the HF repo id for a short alias or an explicit repo id."""
    if "/" in name or ":" in name:
        return name
    return model_name(name)


def load_model(size, dtype=torch.float16, device="mps", weights_dir=None):
    """Load a model. `size` may be a Qwen alias ('4B') or any HF repo id
    (e.g. 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'). `device` is cuda/mps/cpu.
    If `weights_dir` is given, load from that local path instead of hitting the
    HF hub (e.g. the Modal 'model-weights' volume) -- keeps the cache/result
    keys tied to the repo id, not the path. Local checkpoints load with
    trust_remote_code + torch_dtype='auto' (needed for FP4/FP8 quantized
    models like DeepSeek-V4; torch_dtype='auto' keeps fp8/fp4 storage native).
    Returns an eval model (already `.to(device)`) and the tokenizer."""
    # Fall back to the HF hub when the local weights dir doesn't exist (e.g.
    # Qwen3-4B/0.6B not pre-seeded on the Modal volume). Previously this used
    # weights_dir unconditionally, so a missing dir became an invalid repo id
    # and crashed load with HFValidationError.
    name = weights_dir if (weights_dir and os.path.isdir(weights_dir)) \
        else resolve_model(size)
    from_local = os.path.isdir(name)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=from_local)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype if not from_local else "auto",
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=from_local,
        low_cpu_mem_usage=from_local,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tok


def n_layers(model):
    """Discover layer count from a loaded model so arbitrary (incl. huge)
    repos work without a hardcoded map."""
    try:
        return getattr(model.config, "num_hidden_layers")
    except AttributeError:
        # fall back to counting nn layers with block semantics
        for m in model.model.layers:
            return len(model.model.layers)


def default_layers(n):
    step = max(1, n // 8)
    return list(range(0, n, step))[:8] + [n - 1]


def complete(model, tok, prompt, max_new_tokens=40, temperature=None):
    """Greedy (or temp) completion, returns the decoded continuation."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens,
            do_sample=(temperature is not None),
            temperature=temperature, top_p=None,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ----------------------------------------------------------------- logging --
# Named "bench" so bench.py's logging.basicConfig drives these messages too.
LOG = logging.getLogger("bench")


def log(*a):
    LOG.info(" ".join(str(x) for x in a))


# ------------------------------------------------- artifact/cache layout ----
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
#     cache/vec_t{N}_v{M}_l{L}_p{P}_b{B}.npz
#     cache/loop_v{M}_l{L}_p{P}_b{B}.npz
#
# model_slug always comes from resolve_model(alias_or_id) so `0.6B` and
# `Qwen/Qwen3-0.6B` land in the same tree (Qwen__Qwen3-0.6B).
#
# Optional promote: copy the *canonical run file* into RESULTS_DIR under a
# stable name (falls back to latest.json only with a warning), e.g.
#   modal run bench.py ... --promote ruletaker_dsv4_n2k.json
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


def cache_path(cache_dir, max_train, max_val, tag="", layer=None,
               pad_max=PAD_MAX, batch=None):
    """Path under task cache/. Task + model live in the directory tree, not the
    filename. Layer + pad_max MUST stay in the key (stale-vector FIXME); batch
    MUST too: batched-kernel numerics are batch-composition dependent (DSV4
    FP8 moved RuleTaker loop.8 0.7975 -> 0.8375 between b=4 and b=8 on the
    same rows/weights), so vectors or loop scores computed at one batch size
    must never silently replay for a run at another. batch=None keeps the old
    key shape for external/back-compat callers only."""
    ltag = f"_l{layer}" if layer is not None else ""
    btag = f"_b{batch}" if batch is not None else ""
    if tag == "_loop" or (tag and "loop" in tag):
        fname = f"loop_v{max_val}{ltag}_p{pad_max}{btag}.npz"
    else:
        fname = f"vec_t{max_train}_v{max_val}{ltag}_p{pad_max}{btag}{tag}.npz"
    path = os.path.join(cache_dir, fname)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def cache_vectors(max_train, max_val, tag="", cache_dir=CACHE_DIR,
                  layer=None, pad_max=PAD_MAX, batch=None, **arrays):
    """Persist computed hidden vectors + labels so reruns/seed sweeps skip the
    expensive forward passes. Save async stays out of the training loop; a
    blocking save at exit is acceptable here (one-shot per run)."""
    path = cache_path(cache_dir, max_train, max_val, tag,
                      layer=layer, pad_max=pad_max, batch=batch)
    np.savez(path, **arrays)
    log(f"[cache] saved {path}")


def load_cached_vectors(max_train, max_val, keys, tag="", cache_dir=CACHE_DIR,
                        layer=None, pad_max=PAD_MAX, batch=None):
    path = cache_path(cache_dir, max_train, max_val, tag,
                      layer=layer, pad_max=pad_max, batch=batch)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if not all(k in z for k in keys):
            return None
        return {k: z[k] for k in keys}