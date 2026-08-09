"""Shared model/config helpers for the latent-probe experiments."""
import os

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