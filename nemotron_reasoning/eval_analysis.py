from __future__ import annotations

from collections import Counter
from typing import Iterable

from nemotron_reasoning.inference import _summarize_prediction_groups


def index_unique(rows: Iterable[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = row.get("id", "")
        if not row_id:
            raise ValueError(f"{label} contains a row without an id")
        if row_id in indexed:
            raise ValueError(f"{label} contains duplicate id {row_id!r}")
        indexed[row_id] = row
    return indexed


def select_ids(
    rows: list[dict[str, str]],
    ordered_ids: list[str],
    label: str,
) -> list[dict[str, str]]:
    indexed = index_unique(rows, label)
    missing = [row_id for row_id in ordered_ids if row_id not in indexed]
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} requested ids; first={missing[:5]}")
    return [indexed[row_id] for row_id in ordered_ids]


def summarize_prediction_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    scored = [row for row in rows if row.get("correct") in {"True", "False"}]
    correct = sum(row.get("correct") == "True" for row in scored)
    return {
        "row_count": len(rows),
        "scored_count": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else 0.0,
        "method_counts": dict(Counter(row.get("method") or "unknown" for row in rows)),
        "finish_reason_counts": dict(Counter(row.get("finish_reason") or "unknown" for row in rows)),
        "by_task_family": _summarize_prediction_groups(rows, "task_family"),
        "by_task_variant": _summarize_prediction_groups(rows, "task_variant"),
    }


def build_gate_table(
    named_rows: dict[str, list[dict[str, str]]],
    group_key: str,
) -> list[dict[str, object]]:
    group_names = sorted({row.get(group_key, "unknown") or "unknown" for rows in named_rows.values() for row in rows})
    table: list[dict[str, object]] = []
    for group_name in group_names:
        entry: dict[str, object] = {group_key: group_name}
        totals: set[int] = set()
        for name, rows in named_rows.items():
            group_rows = [row for row in rows if (row.get(group_key, "unknown") or "unknown") == group_name]
            scored = [row for row in group_rows if row.get("correct") in {"True", "False"}]
            correct = sum(row.get("correct") == "True" for row in scored)
            totals.add(len(scored))
            entry[name] = {
                "correct": correct,
                "total": len(scored),
                "accuracy": correct / len(scored) if scored else 0.0,
            }
        if len(totals) != 1:
            raise ValueError(f"mismatched totals for {group_key}={group_name!r}: {sorted(totals)}")
        table.append(entry)
    return table


def build_comparison(
    baseline_rows: list[dict[str, str]],
    arm_a_rows: list[dict[str, str]],
    arm_b_rows: list[dict[str, str]],
    full_ids: list[str],
    valid_ids: list[str],
) -> tuple[dict[str, object], dict[str, list[dict[str, str]]]]:
    if len(set(full_ids)) != len(full_ids):
        raise ValueError("full source ids are not unique")
    if len(set(valid_ids)) != len(valid_ids):
        raise ValueError("validation ids are not unique")
    if not set(valid_ids) <= set(full_ids):
        raise ValueError("validation ids are not a subset of full source ids")

    full = {
        "baseline": select_ids(baseline_rows, full_ids, "baseline predictions"),
        "arm_a": select_ids(arm_a_rows, full_ids, "Arm A predictions"),
        "arm_b": select_ids(arm_b_rows, full_ids, "Arm B predictions"),
    }
    heldout = {name: select_ids(rows, valid_ids, f"{name} predictions") for name, rows in full.items()}

    payload: dict[str, object] = {}
    for scope_name, scope_rows in [("full", full), ("heldout", heldout)]:
        payload[scope_name] = {
            "models": {name: summarize_prediction_rows(rows) for name, rows in scope_rows.items()},
            "gate_by_task_family": build_gate_table(scope_rows, "task_family"),
            "gate_by_task_variant": build_gate_table(scope_rows, "task_variant"),
        }
    return payload, heldout
