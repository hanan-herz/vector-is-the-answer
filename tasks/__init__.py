"""Task registry for the latent-probe bench.

Add a new public task by:
  1. Creating ``tasks/<name>.py`` with a ``SPEC`` object implementing TaskSpec
  2. Registering it in ``TASKS`` below
  3. Running ``python bench.py --local --task <name> ...``
"""
from __future__ import annotations

from tasks.arc import SPEC as _arc
from tasks.boolq import SPEC as _boolq
from tasks.ruletaker import SPEC as _ruletaker
from tasks.protocol import TaskSpec

TASKS: dict[str, TaskSpec] = {
    "arc": _arc,
    "boolq": _boolq,
    "ruletaker": _ruletaker,
}


def get_task(name: str | None) -> TaskSpec:
    key = (name or "boolq").lower()
    if key not in TASKS:
        raise ValueError(
            f"unknown task {name!r}; expected one of {sorted(TASKS)}"
        )
    return TASKS[key]


def task_names() -> tuple[str, ...]:
    return tuple(sorted(TASKS))
