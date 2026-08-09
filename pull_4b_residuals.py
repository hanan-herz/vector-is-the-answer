"""Pull the three Qwen3-4B residual caches from the Modal bench-results volume.

They weren't fetched locally (the cloud_bench_cache was copied with
fetch_cache=False). Pull only the residual npz (no weights, no model load) —
confidence smoke is then pure head-forward over cached vectors.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

VOLUME = "bench-results"
OUT = Path("results_4b_residuals")

REQUESTS = {
    "boolq": "Qwen__Qwen3-4B/bench_Qwen_Qwen3-4B_t9427_v3270_l35_p384.npz",
    "ruletaker": "Qwen__Qwen3-4B/ruletaker/cache/vec_t2000_v1000_l35_p384.npz",
    "arc": "Qwen__Qwen3-4B/arc/cache/vec_t1117_v1165_l35_p384.npz",
}


def main():
    vol = modal.Volume.from_name(VOLUME)
    OUT.mkdir(exist_ok=True)
    for name, remote in REQUESTS.items():
        local = OUT / f"{name}.npz"
        if local.exists():
            print(f"[skip] {name}: {local} exists")
            continue
        with open(local, "wb") as out:
            for chunk in vol.read_file(remote):
                out.write(chunk)
        print(f"[got]  {name}: {remote} -> {local} ({local.stat().st_size} bytes)")


if __name__ == "__main__":
    main()