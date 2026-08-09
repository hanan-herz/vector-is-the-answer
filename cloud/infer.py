"""One-shot inference through a checkpoint persisted on the 'model-weights'
volume (no HF re-download). Loads with device_map="auto" across the requested
GPUs and generates a short continuation, then prints input -> output.

deepseek-ai/DeepSeek-V4-Flash-0731 is 167GB (FP4/FP8 MoE), so it needs
>=3 A100-80GB (240GB). Fewer GPUs works but accelerate offloads experts to
CPU and generation gets slow.

IMPORTANT: the checkpoint is FP8-quantized; transformers only keeps FP8 native
on compute capability >= 8.9 (H100/L40S). On A100 (cc 8.0) it dequantizes to
bf16 (~570GB) -> GPU OOM. Use H100 or L40S.

Usage:
    modal run cloud/infer.py --prompt "What is 2+2?"
    modal run cloud/infer.py --gpu H100:3 --max-new-tokens 64
    modal run cloud/infer.py --gpu L40S:4   # 4x48GB = 192GB, also >=8.9
"""
from __future__ import annotations

import os

import torch
import modal

VOLUME_MOUNT = "/weights"
weights_volume = modal.Volume.from_name("model-weights")

app = modal.App("hf-weights-infer")

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "sentencepiece",
                 "kernels==0.15.2")
)


def local_dir(repo_id: str) -> str:
    return os.path.join(VOLUME_MOUNT, repo_id.replace("/", "__"))


@app.function(
    image=infer_image,
    volumes={VOLUME_MOUNT: weights_volume},
    gpu="H100:3",
    memory=64 * 1024,
    timeout=90 * 60,
)
def infer(repo_id: str, prompt: str, max_new_tokens: int = 32) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = local_dir(repo_id)
    print(f"[infer] loading {repo_id} from {path} "
          f"(cuda:{torch.cuda.device_count()})")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.chat_template is None:
        # DeepSeek-V4 ships no chat_template (vLLM uses --tokenizer-mode
        # deepseek_v4); supply the standard DeepSeek template for transformers.
        tok.chat_template = (
            "<｜begin▁of▁sentence｜>{% for message in messages %}"
            "{% if message['role'] == 'user' %}User: {{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}Assistant: "
            "{{ message['content'] }}{% endif %}"
            "{% if loop.last and add_generation_prompt %}"
            "{% if message['role'] == 'user' %}\\n\\nAssistant: {% endif %}"
            "{% else %}\\n\\n{% endif %}{% endfor %}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[infer] loaded. active dev={model.device}")

    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


@app.local_entrypoint()
def main(
    repo: str = "deepseek-ai/DeepSeek-V4-Flash-0731",
    prompt: str = "What is 2+2?",
    max_new_tokens: int = 32,
    gpu: str | None = None,
):
    fn = infer.with_options(gpu=gpu) if gpu else infer
    print(f">>> {prompt}")
    out = fn.remote(repo, prompt, max_new_tokens)
    print(f"<<< {out}")
