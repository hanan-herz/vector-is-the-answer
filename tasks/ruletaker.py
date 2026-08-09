"""RuleTaker task — closed-world rule reasoning with labeled deduction depth.

Source: Hugging Face ``tasksource/ruletaker`` (Clark, Tafjord & Richardson,
IJCAI 2020). Synthetic closed-world rule reasoning with labeled deduction
depth — the property the serial-depth claim wants to bucket on.

Schema on HF (binary NLI-style):
  context  : premise facts + rules
  question : claim to evaluate
  label    : "entailment" | "not entailment"
  config   : depth tag, e.g. "depth-0" .. "depth-5", "depth-3ext",
             "depth-3ext-NatLang", "NatLang"

Normalized row shape matches BoolQ so the shared bench pipeline can reuse
``fmt_example`` / loop / readout arms without forking:

  passage  : context
  question : question
  answer   : bool  (True = entailment)
  depth    : int | None  (primary deduction depth; NatLang -> None)
  config   : original config string
  ext      : bool  (True for depth-3ext / depth-3ext-NatLang)
  natlang  : bool  (True for NatLang / depth-3ext-NatLang)

Splits: train 480_152 / dev 75_872 / test 151_911 (total 707_935).

Loader smoke: ``python -m tasks.ruletaker --split test --max-n 32``
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

# HF hub id used by load_ruletaker
HF_ID = "tasksource/ruletaker"

# Primary depth tags we report as strata. Order is display order.
DEPTH_ORDER = (0, 1, 2, 3, 5)

_DEPTH_RE = re.compile(r"^depth-(\d+)")


def parse_depth(config: str) -> int | None:
    """Extract the primary deduction depth from a RuleTaker config tag.

    ``depth-0``..``depth-5`` and ``depth-3ext*`` return an int; pure
    ``NatLang`` (hand-authored, no depth label) returns None.
    """
    if not config:
        return None
    m = _DEPTH_RE.match(config)
    if m:
        return int(m.group(1))
    return None


def is_ext(config: str) -> bool:
    return "ext" in (config or "")


def is_natlang(config: str) -> bool:
    return "NatLang" in (config or "")


def normalize_row(r: dict) -> dict:
    """Map one HF row to the BoolQ-compatible shape used by bench.py."""
    config = r.get("config") or ""
    label = r["label"]
    if label not in ("entailment", "not entailment"):
        raise ValueError(f"unexpected RuleTaker label: {label!r}")
    return {
        "passage": r["context"],
        "question": r["question"],
        "answer": label == "entailment",
        "depth": parse_depth(config),
        "config": config,
        "ext": is_ext(config),
        "natlang": is_natlang(config),
    }


def load_ruletaker(
    split: str = "test",
    max_n: int | None = None,
    seed: int = 0,
    depths: Iterable[int] | None = None,
    include_natlang: bool = True,
    include_ext: bool = True,
    *,
    streaming: bool = False,
) -> list[dict]:
    """Load and normalize RuleTaker rows from Hugging Face.

    Args:
      split: "train" | "dev" | "test"
      max_n: if set, shuffle (seed) and take the first max_n after filtering
      seed: RNG seed for the subsample shuffle
      depths: if set, keep only rows whose primary depth is in this set
              (NatLang rows have depth=None and are dropped unless
              include_natlang and depths is None)
      include_natlang: keep pure NatLang / *NatLang configs
      include_ext: keep depth-3ext / depth-3ext-NatLang configs
      streaming: pass through to datasets (rarely useful; default False)

    Returns:
      list of normalized dicts (see module docstring).
    """
    from datasets import load_dataset
    import numpy as np

    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be train/dev/test, got {split!r}")

    ds = load_dataset(HF_ID, split=split, streaming=streaming)
    depth_set = set(depths) if depths is not None else None
    needs_filter = (
        depth_set is not None or not include_ext or not include_natlang
    )

    # Fast path: no config filters and a small max_n — shuffle+select on the
    # Arrow table so we never materialize 480k Python dicts for a smoke run.
    if not needs_filter and not streaming and max_n is not None:
        n = min(max_n, len(ds))
        ds = ds.shuffle(seed=seed).select(range(n))
        return [normalize_row(dict(r)) for r in ds]

    rows: list[dict] = []
    for r in ds:
        row = normalize_row(dict(r))
        if not include_ext and row["ext"]:
            continue
        if not include_natlang and row["natlang"]:
            continue
        if depth_set is not None:
            if row["depth"] is None or row["depth"] not in depth_set:
                continue
        rows.append(row)

    if max_n is not None and max_n < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))[:max_n]
        rows = [rows[i] for i in idx]

    return rows


def depth_counts(rows: list[dict]) -> dict:
    """Return {depth_key: n} for reporting. depth_key is int or 'NatLang'."""
    c: Counter = Counter()
    for r in rows:
        key = r["depth"] if r["depth"] is not None else "NatLang"
        c[key] += 1
    return dict(c)


def depth_buckets(rows: list[dict]) -> dict[str, list[int]]:
    """Map depth key -> list of row indices (for stratum slicing).

    Keys are str(depth) for 0/1/2/3/5 and 'NatLang' for unlabeled rows.
    """
    buckets: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        key = str(r["depth"]) if r["depth"] is not None else "NatLang"
        buckets.setdefault(key, []).append(i)
    # stable display order: DEPTH_ORDER then NatLang if present
    ordered: dict[str, list[int]] = {}
    for d in DEPTH_ORDER:
        k = str(d)
        if k in buckets:
            ordered[k] = buckets[k]
    if "NatLang" in buckets:
        ordered["NatLang"] = buckets["NatLang"]
    # any unexpected depths
    for k, v in buckets.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def label_balance(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "pct_true": 0.0, "n_true": 0, "n_false": 0}
    n_true = sum(1 for r in rows if r["answer"])
    return {
        "n": n,
        "n_true": n_true,
        "n_false": n - n_true,
        "pct_true": n_true / n,
    }


def summarize(rows: list[dict]) -> dict:
    """Compact summary for logging / results meta."""
    return {
        "n": len(rows),
        "label": label_balance(rows),
        "depth": depth_counts(rows),
        "config": dict(Counter(r["config"] for r in rows)),
    }


class RuleTakerTask:
    name = "ruletaker"
    answer_set = ("Yes", "No")
    # Full train is ~480k — defaults keep a first run tractable.
    default_max_train: int | None = 10_000
    default_max_val: int | None = 4_000

    def load(
        self,
        max_train: int | None,
        max_val: int | None,
        rng: Any,
    ) -> tuple[list[dict], list[dict], dict, int, int]:
        del rng  # subsample seeds are fixed in load_ruletaker for cache stability
        if max_train is None:
            max_train = self.default_max_train
        if max_val is None:
            max_val = self.default_max_val
        train = load_ruletaker(split="train", max_n=max_train, seed=0)
        # Official test split for eval; depth strata reported via strata().
        val = load_ruletaker(split="test", max_n=max_val, seed=1)
        meta = {
            "task": self.name,
            "hf_id": HF_ID,
            "val_split": "test",
            "train_summary": summarize(train),
            "val_summary": summarize(val),
        }
        return train, val, meta, max_train, max_val

    def strata(
        self, val_rows: list[dict]
    ) -> dict[str, dict[str, list[int]]]:
        # Serial-depth claim axis: readout should stay flat (or degrade
        # gracefully) across depth-0..depth-5, not cliff at d>=3.
        return {"depth": depth_buckets(val_rows)}


SPEC = RuleTakerTask()


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="RuleTaker loader smoke test")
    ap.add_argument("--split", default="test", choices=("train", "dev", "test"))
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depths", default=None,
                    help="comma-separated depths to keep, e.g. 0,1,2,3,5")
    ap.add_argument("--no-natlang", action="store_true")
    ap.add_argument("--no-ext", action="store_true")
    args = ap.parse_args()

    depths = None
    if args.depths:
        depths = [int(x) for x in args.depths.split(",") if x.strip() != ""]

    rows = load_ruletaker(
        split=args.split,
        max_n=args.max_n,
        seed=args.seed,
        depths=depths,
        include_natlang=not args.no_natlang,
        include_ext=not args.no_ext,
    )
    print(json.dumps(summarize(rows), indent=2, default=str))
    if rows:
        print("--- sample ---")
        s = rows[0]
        print(f"depth={s['depth']} config={s['config']} answer={s['answer']}")
        print(f"Q: {s['question']}")
        print(f"P: {s['passage'][:200]}...")
