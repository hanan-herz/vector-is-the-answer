"""ARC-Challenge task — grade-school science MC (allenai/ai2_arc).

Source: Clark et al. 2018, *Think you have Solved Question Answering? Try ARC*.
Challenge subset = questions that both retrieval and co-occurrence baselines
missed. Answers depend on parametric science knowledge (no passage).

Normalized row shape (multi-class extension of the BoolQ core):

  passage  : ""  (no context document; knowledge is in the weights)
  question : stem + listed choices, e.g. "...?\\n(A) ...\\n(B) ..."
  answer   : int class index in 0..3 matching answer_set order
  answer_key : original letter ("A".."D")
  id       : HF id

Only 4-choice items with mappable A–D keys are kept (~99% of Challenge).
Chance baseline = 0.25.

Loader smoke: ``python -m tasks.arc --split test --max-n 8``
"""
from __future__ import annotations

from collections import Counter
from typing import Any

HF_ID = "allenai/ai2_arc"
HF_CONFIG = "ARC-Challenge"
ANSWER_SET = ("A", "B", "C", "D")

# Map numeric keys used on a minority of rows onto letters.
_KEY_MAP = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
}


def _normalize_key(raw: str) -> str | None:
    if raw is None:
        return None
    return _KEY_MAP.get(str(raw).strip().upper())


def format_choices(labels: list[str], texts: list[str]) -> str:
    lines = []
    for lab, txt in zip(labels, texts):
        letter = _normalize_key(lab) or str(lab)
        lines.append(f"({letter}) {txt}")
    return "\n".join(lines)


def normalize_row(r: dict) -> dict | None:
    """Map one HF row to the multi-class bench shape, or None if dropped."""
    labels = list(r["choices"]["label"])
    texts = list(r["choices"]["text"])
    if len(labels) != 4 or len(texts) != 4:
        return None
    key = _normalize_key(r["answerKey"])
    if key not in ANSWER_SET:
        return None
    # Ensure choice labels are A–D (some rows use 1–4).
    letter_labels = [_normalize_key(x) for x in labels]
    if letter_labels != list(ANSWER_SET):
        # Re-order / re-label if HF used 1–4 in order.
        if letter_labels == ["A", "B", "C", "D"]:
            pass
        elif labels == ["1", "2", "3", "4"] or letter_labels == list(ANSWER_SET):
            letter_labels = list(ANSWER_SET)
        else:
            # Unexpected label order — zip by mapped letter.
            by_letter = {_normalize_key(l): t for l, t in zip(labels, texts)}
            if not all(k in by_letter for k in ANSWER_SET):
                return None
            texts = [by_letter[k] for k in ANSWER_SET]
            letter_labels = list(ANSWER_SET)
    class_idx = ANSWER_SET.index(key)
    stem = r["question"].strip()
    choices_block = format_choices(letter_labels, texts)
    return {
        "passage": "",
        "question": f"{stem}\n{choices_block}",
        "answer": class_idx,
        "answer_key": key,
        "id": r.get("id", ""),
        "choices": list(texts),
    }


def load_arc(
    split: str = "test",
    max_n: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Load ARC-Challenge rows from Hugging Face and normalize.

    Args:
      split: "train" | "validation" | "test"
      max_n: optional shuffle+truncate after filtering
      seed: RNG seed for the subsample shuffle
    """
    from datasets import load_dataset
    import numpy as np

    if split not in ("train", "validation", "test"):
        raise ValueError(f"split must be train/validation/test, got {split!r}")

    ds = load_dataset(HF_ID, HF_CONFIG, split=split)
    rows: list[dict] = []
    for r in ds:
        row = normalize_row(dict(r))
        if row is not None:
            rows.append(row)

    if max_n is not None and max_n < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))[:max_n]
        rows = [rows[i] for i in idx]
    return rows


def label_counts(rows: list[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        c[ANSWER_SET[int(r["answer"])]] += 1
    return {k: int(c.get(k, 0)) for k in ANSWER_SET}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    counts = label_counts(rows)
    return {
        "n": n,
        "labels": counts,
        "majority": (max(counts.values()) / n) if n else 0.0,
        "chance": 1.0 / len(ANSWER_SET),
    }


class ArcChallengeTask:
    name = "arc"
    answer_set = ANSWER_SET
    # Full Challenge train/test fit easily on one GPU pass.
    default_max_train: int | None = None
    default_max_val: int | None = None

    def load(
        self,
        max_train: int | None,
        max_val: int | None,
        rng: Any,
    ) -> tuple[list[dict], list[dict], dict, int, int]:
        del rng  # fixed seeds in load_arc for cache stability
        train = load_arc(split="train", max_n=max_train, seed=0)
        # Official test split for eval (same discipline as RuleTaker).
        val = load_arc(split="test", max_n=max_val, seed=1)
        if max_train is None:
            max_train = len(train)
        if max_val is None:
            max_val = len(val)
        meta = {
            "task": self.name,
            "hf_id": HF_ID,
            "hf_config": HF_CONFIG,
            "val_split": "test",
            "answer_set": list(self.answer_set),
            "n_classes": len(self.answer_set),
            "train_summary": summarize(train),
            "val_summary": summarize(val),
            "filter": "4-choice A-D only",
        }
        return train, val, meta, max_train, max_val

    def strata(self, val_rows: list[dict]) -> dict[str, dict[str, list[int]]]:
        # Per-letter buckets (class balance / accuracy by answer key).
        buckets: dict[str, list[int]] = {k: [] for k in ANSWER_SET}
        for i, r in enumerate(val_rows):
            buckets[ANSWER_SET[int(r["answer"])]].append(i)
        return {"answer_key": {k: v for k, v in buckets.items() if v}}


SPEC = ArcChallengeTask()


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="ARC-Challenge loader smoke test")
    ap.add_argument("--split", default="test",
                    choices=("train", "validation", "test"))
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_arc(split=args.split, max_n=args.max_n, seed=args.seed)
    print(json.dumps(summarize(rows), indent=2))
    if rows:
        print("--- sample ---")
        s = rows[0]
        print(f"id={s['id']} answer={s['answer']} key={s['answer_key']}")
        print(s["question"][:400])
        print("...")
