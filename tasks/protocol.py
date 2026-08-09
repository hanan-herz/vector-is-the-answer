"""Shared task contract for the latent-probe bench.

Every task normalizes rows to a shared core so the pipeline
(fmt_example / hidden vectors / readout / loop) stays mostly unforked:

  passage  : str   (may be "" when knowledge is parametric, e.g. ARC)
  question : str   (may include listed MC choices)
  answer   : bool  (binary tasks: True/False)
             | int (multi-class: index into ``answer_set``)

``answer_set`` is the closed vocabulary for the fair loop next-token read
(e.g. ``("Yes","No")`` or ``("A","B","C","D")``). Binary tasks keep bool
labels; multi-class tasks use integer class indices.

Optional per-task fields (depth, config, answer_key, ...) are free; use
``strata`` to expose index buckets the bench reports after length/label strata.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskSpec(Protocol):
    """Minimal interface a task module must satisfy."""

    name: str
    # Closed answer vocabulary for the fair loop (next-token log-prob).
    answer_set: tuple[str, ...]
    # Applied when the caller passes max_train / max_val as None.
    default_max_train: int | None
    default_max_val: int | None

    def load(
        self,
        max_train: int | None,
        max_val: int | None,
        rng: Any,
    ) -> tuple[list[dict], list[dict], dict, int, int]:
        """Return (train, val, meta, max_train, max_val).

        ``meta`` is merged into result["meta"] (task name, hf_id, summaries…).
        ``max_train`` / ``max_val`` are the resolved caps after defaults.
        """
        ...

    def strata(
        self, val_rows: list[dict]
    ) -> dict[str, dict[str, list[int]]] | None:
        """Optional extra eval axes: {axis_name: {bucket_key: [val indices]}}.

        Example RuleTaker: ``{"depth": {"0": [...], "3": [...], "NatLang": [...]}}``
        → stored as ``result["stratum_depth"]``. Return None to skip.
        """
        ...
