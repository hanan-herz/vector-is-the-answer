"""BoolQ task — binary reading comprehension (google/boolq)."""
from __future__ import annotations

from typing import Any


class BoolQTask:
    name = "boolq"
    answer_set = ("Yes", "No")
    # None → use full split sizes after load.
    default_max_train: int | None = None
    default_max_val: int | None = None

    def load(
        self,
        max_train: int | None,
        max_val: int | None,
        rng: Any,
    ) -> tuple[list[dict], list[dict], dict, int, int]:
        from datasets import load_dataset

        ds = load_dataset("google/boolq")
        train = [dict(r) for r in ds["train"]]
        val = [dict(r) for r in ds["validation"]]
        rng.shuffle(train)
        if max_train is None:
            max_train = (
                self.default_max_train
                if self.default_max_train is not None
                else len(train)
            )
        if max_val is None:
            max_val = (
                self.default_max_val
                if self.default_max_val is not None
                else len(val)
            )
        train = train[:max_train]
        val = val[:max_val]
        meta = {
            "task": self.name,
            "hf_id": "google/boolq",
            "val_split": "validation",
        }
        return train, val, meta, max_train, max_val

    def strata(self, val_rows: list[dict]) -> None:
        return None


SPEC = BoolQTask()
