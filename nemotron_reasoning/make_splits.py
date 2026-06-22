from __future__ import annotations

import random
from collections import defaultdict

from nemotron_reasoning.task_types import task_variant


def split_train_valid(
    rows: list[dict[str, str]],
    valid_fraction: float = 0.2,
    seed: int = 12345,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create the task-variant-stratified split used by the experiment."""
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[task_variant(row.get("prompt"))].append(row)

    train_rows: list[dict[str, str]] = []
    valid_rows: list[dict[str, str]] = []
    for variant in sorted(groups):
        group_rows = list(groups[variant])
        rng.shuffle(group_rows)
        if len(group_rows) == 1:
            valid_count = 0
        else:
            valid_count = max(1, round(len(group_rows) * valid_fraction))
            valid_count = min(valid_count, len(group_rows) - 1)
        valid_rows.extend(group_rows[:valid_count])
        train_rows.extend(group_rows[valid_count:])

    train_rows.sort(key=lambda row: str(row.get("id", "")))
    valid_rows.sort(key=lambda row: str(row.get("id", "")))
    return train_rows, valid_rows
