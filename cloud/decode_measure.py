"""Measure whether a greedy decode from the bench "Answer:" prompt
self-terminates (hits EOS) or runs to the cap — the number behind
the "≥300 tokens" claim in RESULTS.md ("FLOPs, honestly"), now at DeepSeek-V4 scale.

Uses the SAME conditioning as bench.py's loop arm (`fmt_example`: a plain
"Passage: ...\nQuestion: ...\nAnswer:" string, no chat template), greedy
decode, so a non-self-terminating run here means the Ext-6-style decoding-loop
FLOP win (≈2·params/token) applies to a frontier MoE too, not just 0.6B.

Two-stage protocol in ONE container (model load dominates; never re-load):
stage 1 decodes all rows at `caps[0]` (default 200); any row that hits that
cap is re-decoded at the next cap (default 300) from a fresh KV cache. Mirrors
the 0.6B measurement (25/25 hit 200, 8/8 hit 300) without the load cost of a
second container.

Run:
    modal run cloud/decode_measure.py --samples 8 --caps 200,300
"""
from __future__ import annotations

import os

import modal

VOLUME_MOUNT = "/weights"
WEIGHTS_VOLUME = "model-weights"
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

app = modal.App("decode-measure")

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.46", "datasets", "accelerate",
                 "sentencepiece", "protobuf", "kernels==0.15.2")
    .env({
        "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
)


def local_dir(repo_id: str) -> str:
    return os.path.join(VOLUME_MOUNT, repo_id.replace("/", "__"))


def fmt_example(r) -> str:
    return f"Passage: {r['passage']}\nQuestion: {r['question']}\nAnswer:"


def first_eos_of(cont, eos):
    return [int(l[0].item()) if (l := (cont[i] == eos).nonzero(as_tuple=True)[0]).numel()
            else None for i in range(cont.size(0))]


@app.function(
    image=infer_image,
    volumes={VOLUME_MOUNT: weights_volume},
    gpu="b300",
    cpu=32.0,
    memory=32768,
    timeout=60 * 60,
)
def measure(repo_id: str, n: int, caps: list[int]) -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = local_dir(repo_id)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # Batched generation from decoder-only models needs LEFT padding so the real
    # tokens sit at the end and generation starts from them (right-padding makes
    # the model attend to pad tokens and continue from the wrong position).
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[decode_measure] loaded {repo_id} from {path} "
          f"(cuda:{torch.cuda.device_count()})")

    ds = load_dataset("google/boolq")
    rows = [dict(r) for r in ds["validation"]][:n]
    texts = [fmt_example(r) for r in rows]

    stages = {}
    todo = list(range(n))
    for cap in caps:
        if not todo:
            break
        sub = [texts[i] for i in todo]
        enc = tok(sub, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=cap, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        cont = out[:, enc["input_ids"].shape[1]:]
        eos = first_eos_of(cont, tok.eos_token_id)
        for i, e in zip(todo, eos):
            stages[i] = {"cap": cap,
                         "first_eos": e,
                         "hit_cap": e is None}
            print(f"  row {i} @ cap {cap}: "
                  f"{('EOS @ ' + str(e)) if e is not None else 'CAP'}")
        todo = [i for i in todo if stages[i]["hit_cap"]]
        if todo:
            print(f"  stage {cap}: {len(todo)} still rambling -> next cap "
                  f"{[c for c in caps if c > cap] or 'DONE'}")

    n_term = sum(1 for s in stages.values() if not s["hit_cap"])
    print(f"[decode_measure] n={n} self_terminated={n_term}/{n} "
          f"hit_final_cap={sum(1 for s in stages.values() if s['hit_cap'])}/{n} "
          f"(caps={caps})")
    return {"n": n, "caps": caps,
            "self_terminated": n_term,
            "hit_final_cap": sum(1 for s in stages.values() if s["hit_cap"]),
            "rows": {str(i): s for i, s in stages.items()}}


@app.local_entrypoint()
def main(
    repo: str = MODEL,
    samples: int = 8,
    caps: str = "200,300",
):
    cap_list = [int(x) for x in caps.split(",")]
    res = measure.remote(repo, samples, cap_list)
    print(f"\n=== self_terminated={res['self_terminated']}/{res['n']} "
          f"hit_final_cap={res['hit_final_cap']}/{res['n']} "
          f"(caps={res['caps']})")
