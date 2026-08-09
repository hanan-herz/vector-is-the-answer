"""Run the 'Paris is in' completion across all model sizes (greedy).
This is the quick qualitative probe from the md — does the decoded readout
expose the latent geopolitical relation (France)?"""
import argparse

import torch
from common import SIZES, complete, load_model

PROMPTS = ["Paris is in"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(SIZES),
                    help="comma-separated model sizes to run")
    ap.add_argument("--temp", type=float, default=None,
                    help="sampling temperature (default greedy)")
    ap.add_argument("--tokens", type=int, default=40)
    args = ap.parse_args()
    sizes = [s for s in args.sizes.split(",") if s in SIZES]

    for size in sizes:
        print(f"\n===== Qwen3-{size} =====")
        model, tok = load_model(size, dtype=torch.float16)
        try:
            for p in PROMPTS:
                out = complete(model, tok, p,
                               max_new_tokens=args.tokens, temperature=args.temp)
                print(f"  {p!r} -> {out!r}")
        finally:
            del model, tok
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()