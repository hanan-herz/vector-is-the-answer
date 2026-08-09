"""Persist an open-weights model repo (e.g. deepseek-ai/DeepSeek-V4-Flash-0731,
~167GB / 48 safetensor shards) into a long-lived Modal Volume once, so future
bench/test runs load weights from the volume instead of re-downloading from
Hugging Face every time.

Weights are stored at /weights/<repo_id with "/" -> "__"> on the "model-weights"
volume. A later test loads them with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/weights/deepseek__DeepSeek-V4-Flash-0731")
    model = AutoModelForCausalLM.from_pretrained(
        "/weights/deepseek__DeepSeek-V4-Flash-0731",
        trust_remote_code=True, device_map="auto", torch_dtype="auto",
    )

Usage (script mode, run by file path):

    modal run cloud/weights.py --repo deepseek-ai/DeepSeek-V4-Flash-0731
    modal run cloud/weights.py --repo deepseek-ai/DeepSeek-V4-Flash-0731 --revision main

    # only check what is already stored (no download):
    modal run cloud/weights.py --repo deepseek-ai/DeepSeek-V4-Flash-0731 --verify-only

    # inspect / clean up from the volume via the CLI:
    modal volume ls model-weights
    modal volume rm model-weights deepseek__DeepSeek-V4-Flash-0731
"""
from __future__ import annotations

import json
import os

import modal

VOLUME_MOUNT = "/weights"
WEIGHTS_VOLUME = "model-weights"
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)

# xet backend for fast parallel transfers; hf_transfer as fallback.
HUB_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[cli,xet]", "hf_transfer")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

app = modal.App("hf-weights-store")


def volume_dir(repo_id: str) -> str:
    return os.path.join(VOLUME_MOUNT, repo_id.replace("/", "__"))


@app.function(
    image=HUB_IMAGE,
    volumes={VOLUME_MOUNT: weights_volume},
    timeout=3 * 60 * 60,
    cpu=8.0,
    memory=16 * 1024,
)
def download_weights(repo_id: str, revision: str | None = None) -> dict:
    """snapshot_download the repo straight into the mounted volume, then commit.
    Idempotent: existing files are skipped, so a rerun resumes/finishes."""
    from huggingface_hub import snapshot_download

    local_dir = volume_dir(repo_id)
    os.makedirs(local_dir, exist_ok=True)
    import huggingface_hub as hf

    print(f"[download] hub={hf.__version__} {repo_id}@{revision or 'main'} -> {local_dir}")
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
    weights_volume.commit()
    print(f"[download] committed {local_dir}")
    return verify_local(repo_id)


@app.function(
    image=HUB_IMAGE,
    volumes={VOLUME_MOUNT: weights_volume},
    timeout=15 * 60,
    cpu=1.0,
    memory=512,
)
def verify_weights(repo_id: str) -> dict:
    return verify_local(repo_id)


def verify_local(repo_id: str) -> dict:
    """File-level integrity check: every shard named in
    model.safetensors.index.json exists and is non-empty, config/tokenizer are
    present. Reports total bytes so ~166.9GB is easy to eyeball."""
    local_dir = volume_dir(repo_id)
    if not os.path.isdir(local_dir):
        return {"ok": False, "error": f"{local_dir} not present on volume",
                "volume_dir": local_dir, "repo_id": repo_id}

    total = 0
    n_shards = 0
    zero = []
    for root, dirs, files in os.walk(local_dir):
        for name in files:
            p = os.path.join(root, name)
            sz = os.path.getsize(p)
            total += sz
            if name.startswith("model-") and name.endswith(".safetensors"):
                n_shards += 1
                if sz == 0:
                    zero.append(os.path.relpath(p, local_dir))

    expected = []
    idx = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            expected = sorted(set(json.load(fh)["weight_map"].values()))
    missing = [s for s in expected
               if not os.path.exists(os.path.join(local_dir, s))]

    ok = (
        not missing and not zero
        and os.path.exists(os.path.join(local_dir, "config.json"))
        and os.path.exists(os.path.join(local_dir, "tokenizer.json"))
    )
    info = {
        "ok": ok,
        "repo_id": repo_id,
        "volume_dir": local_dir,
        "shards": n_shards,
        "shards_expected": len(expected),
        "missing_shards": missing,
        "zero_shards": zero,
        "total_bytes": total,
        "total_gib": round(total / 2 ** 30, 2),
    }
    print(json.dumps(info, indent=2))
    return info


@app.local_entrypoint()
def main(
    repo: str = "deepseek-ai/DeepSeek-V4-Flash-0731",
    revision: str | None = None,
    verify_only: bool = False,
):
    info = (verify_weights.remote(repo) if verify_only
            else download_weights.remote(repo, revision))
    ok = info.get("ok", False)
    if "error" in info:
        print(f"\nverify: {info['error']} ({info['volume_dir']})")
        raise SystemExit(1)
    print(f"\nverify {'PASS' if ok else 'FAIL'}: {repo} -> {info['volume_dir']} "
          f"({info['total_gib']} GiB, {info['shards']}/{info['shards_expected']} shards)")
    if not ok:
        raise SystemExit(1)
