"""Measure greedy-decode self-termination on DeepSeek-V4-Flash via the public
OpenRouter API (proxy of the bench's own model family) instead of a Modal
container. Uses the SAME conditioning as bench.py's loop arm (`fmt_example`:
plain "Passage: ...\nQuestion: ...\nAnswer:", no chat template), temperature=0
to approximate greedy, and reads finish_reason to classify stop vs length-cap.
"""
import json
import os
import re
import sys
import time

import requests

ROUTER = "https://openrouter.ai/api/v1/chat/completions"


def read_key():
    p = os.path.expanduser("~/.grok/config.toml")
    for line in open(p):
        if "api_key" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no api_key in ~/.grok/config.toml")


def fmt_example(r):
    return f"Passage: {r['passage']}\nQuestion: {r['question']}\nAnswer:"


def run(prompt, key, model, cap):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": cap,
        # The served model is a reasoning variant (returns `reasoning` + empty
        # content). Disable it so the budget goes to the visible answer and we
        # approximate a raw greedy decode from the bench's "Answer:" prompt.
        "reasoning": {"enabled": False},
    }
    r = requests.post(ROUTER, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }, json=body, timeout=120)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    if not content:
        print(f"[debug] empty content: finish={ch.get('finish_reason')} "
              f"msg_keys={list(msg.keys())}")
    return ch["finish_reason"], content.split()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    model = os.environ.get("DM_MODEL", "~deepseek/deepseek-v4-flash-latest")
    key = read_key()

    from datasets import load_dataset
    ds = load_dataset("google/boolq")
    rows = [dict(r) for r in ds["validation"]][:n]

    term, capped = 0, 0
    for i, r in enumerate(rows):
        prompt = fmt_example(r)
        fr, words = run(prompt, key, model, cap)
        is_cap = fr == "length"
        term += not is_cap
        capped += is_cap
        print(f"row {i}: finish={fr:<7} words={len(words):<4} "
              f"{'CAP ' + str(cap) if is_cap else 'self-terminated'}")
        time.sleep(1)

    print(f"\n=== self_terminated={term}/{n} hit_cap={capped}/{n} (cap={cap}) "
          f"model={model}")


if __name__ == "__main__":
    main()
