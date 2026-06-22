from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nemotron_reasoning.io_utils import read_csv_rows, read_jsonl, write_csv_rows, write_json, write_jsonl
from nemotron_reasoning.make_splits import split_train_valid
from nemotron_reasoning.prompts import kaggle_user_content
from nemotron_reasoning.symbol_cipher import canonical_op_name, parse_puzzle
from nemotron_reasoning.task_types import task_family, task_variant
from nemotron_reasoning.trace_surgeon import (
    DEDUCE_PRIOR_TOP4,
    GUESS_PRIOR,
    PrefixVerificationError,
    TraceVerificationError,
    add_conversion_header_byte_continuity,
    approximate_token_count,
    complete_prefix_info,
    enumerate_witness_programs,
    parse_verified_prefix,
    render_arm_a,
    render_arm_b,
    trace_completion_with_prefix,
    undefined_letter_references,
    verify_prefix_byte_continuity,
    verify_rendered_trace,
)

OTHER_FAMILIES = ["roman_numeral", "gravity", "unit_conversion", "cipher_text", "bit_manipulation"]
CSV_FIELDS = ["id", "prompt", "prefix", "completion"]


def truthy(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def token_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    lengths = [approximate_token_count(row["completion"]) for row in rows]
    return {
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "max": max(lengths) if lengths else 0,
    }


def operation_prior(records: list[dict[str, Any]], subtype: str) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("status") != "solved" or not record.get("matches_gold"):
            continue
        if record.get("subtype") != subtype:
            continue
        query = record.get("query_op", "")
        op = canonical_op_name((record.get("ops") or {}).get(query, ""))
        if op:
            counts[op] += 1
    return [op for op, _ in counts.most_common()]


def record_query_op(record: dict[str, Any]) -> str:
    return canonical_op_name((record.get("ops") or {}).get(record.get("query_op", ""), ""))


def ambiguous_included(
    puzzle,
    record: dict[str, Any],
    deduce_rank: dict[str, int],
) -> tuple[bool, dict[str, Any]]:
    gold_op = record_query_op(record)
    gold_rank = deduce_rank.get(gold_op, 10_000)
    witnesses = enumerate_witness_programs(puzzle, limit=24, solution_limit=256)
    alternatives = [
        {"answer": witness.answer, "query_op": witness.query_op, "mode": witness.mode}
        for witness in witnesses
        if witness.answer != puzzle.gold
    ]
    alt_ops = {canonical_op_name(item["query_op"]) for item in alternatives}
    include = bool(alternatives) and all(gold_rank < deduce_rank.get(op, 10_000) for op in alt_ops)
    return include, {"witnesses": alternatives[:8], "gold_rank": gold_rank, "alt_ops": sorted(alt_ops)}


def inclusion_decision(
    puzzle,
    record: dict[str, Any],
    baseline_correct: bool,
    deduce_rank: dict[str, int],
    guess_prior: list[str],
) -> tuple[bool, str, dict[str, Any]]:
    if record.get("status") != "solved" or not record.get("matches_gold"):
        return False, "solver_not_solved_match", {}
    if baseline_correct:
        return False, "baseline_already_correct", {}

    gold_op = record_query_op(record)
    subtype = record.get("subtype")
    if subtype == "deduce":
        status = record.get("uniqueness_status")
        if status == "unique":
            return True, "deduce_unique", {}
        if status == "ambiguous":
            include, meta = ambiguous_included(puzzle, record, deduce_rank)
            return (True, "deduce_ambiguous_prior", meta) if include else (False, "ambiguous_gold_not_prior_winner", meta)
        if status == "unknown":
            if gold_op in DEDUCE_PRIOR_TOP4:
                return True, "deduce_unknown_top4", {"gold_op": gold_op}
            return False, "unknown_gold_not_top4", {"gold_op": gold_op}
        return False, f"deduce_bad_uniqueness_status_{status}", {}

    if subtype == "guess":
        top = guess_prior[0] if guess_prior else GUESS_PRIOR[0]
        if gold_op == top:
            return True, "guess_prior_top", {"gold_op": gold_op, "prior_top": top}
        return False, "guess_gold_not_prior_top", {"gold_op": gold_op, "prior_top": top}

    return False, f"unsupported_subtype_{subtype}", {}


def canonical_split_ids(rows: list[dict[str, str]]) -> tuple[set[str], set[str]]:
    train_rows, valid_rows = split_train_valid(rows, valid_fraction=0.2, seed=12345)
    train_ids = {row["id"] for row in train_rows}
    valid_ids = {row["id"] for row in valid_rows}
    if len(train_rows) != 7601 or len(valid_rows) != 1899:
        raise RuntimeError(f"canonical split count mismatch: train={len(train_rows)} valid={len(valid_rows)}")
    overlap = train_ids & valid_ids
    if overlap:
        raise RuntimeError(f"canonical split overlap: {sorted(overlap)[:5]}")
    return train_ids, valid_ids


def build_anchor_rows(
    baseline_rows: list[dict[str, str]],
    train_by_id: dict[str, dict[str, str]],
    train_ids: set[str],
    seed: int,
    per_family: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_rows:
        row_id = row.get("id", "")
        if row_id not in train_ids:
            continue
        if not truthy(row.get("correct")):
            continue
        family = row.get("task_family") or task_family(train_by_id.get(row_id, {}).get("prompt", ""))
        if family not in OTHER_FAMILIES:
            continue
        if not row.get("prediction"):
            continue
        by_family[family].append(row)

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for family in OTHER_FAMILIES:
        candidates = list(by_family.get(family, []))
        chosen = rng.sample(candidates, min(per_family, len(candidates))) if candidates else []
        counts[family] = len(chosen)
        for row in chosen:
            source = train_by_id[row["id"]]
            selected.append(
                {
                    "id": row["id"],
                    "prompt": source["prompt"],
                    "prefix": "",
                    "completion": row["prediction"],
                    "source": "anchor",
                    "split": "train",
                    "task_family": family,
                    "inclusion_category": "anchor_baseline_correct",
                }
            )
    rng.shuffle(selected)
    return selected, counts


def composed_prompt_stub(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "<STUB_CHAT_TEMPLATE_BEGIN>",
            "USER_CONTENT:",
            kaggle_user_content(row["prompt"]),
            "ASSISTANT_GENERATION_PROMPT:",
            "<STUB_ASSISTANT_PREFIX_BEGIN>",
            row.get("prefix", ""),
            "<STUB_ASSISTANT_PREFIX_END>",
            "<STUB_CHAT_TEMPLATE_END>",
        ]
    )


def write_samples(path: Path, arm_a_rows: list[dict[str, Any]], arm_b_rows: list[dict[str, Any]]) -> None:
    def pick_many(rows: list[dict[str, Any]], mode: str, count: int) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        for row in rows:
            if row.get("source") == "eq_symbol" and row.get("mode") == mode:
                picked.append(row)
            if len(picked) >= count:
                break
        return picked

    samples = [
        ("arm_a standard", row) for row in pick_many(arm_a_rows, "standard", 2)
    ]
    samples.extend(("arm_a little_endian", row) for row in pick_many(arm_a_rows, "little_endian", 1))
    samples.extend(("arm_b", row) for row in arm_b_rows if row.get("source") == "eq_symbol")
    samples = samples[:4]

    chunks: list[str] = []
    for label, row in samples:
        chunks.append(
            "\n".join(
                [
                    f"===== {label} id={row['id']} split={row.get('split', '')} mode={row.get('mode', '')} =====",
                    "RAW PROMPT:",
                    row["prompt"],
                    "PREFIX:",
                    row.get("prefix", ""),
                    "COMPOSED PROMPT VIEW:",
                    composed_prompt_stub(row),
                    "COMPLETION:",
                    row["completion"],
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")


def build_corpora(
    train_csv: str = "data/train.csv",
    solver_report: str = "runs/eqsym_surgery_001/solver/solver_report.jsonl",
    baseline_predictions: str = "runs/kien_tinker_086_baseline/predictions/train_full_vllm.csv",
    trace_dir: str = "runs/eqsym_surgery_001/traces",
    arm_a_train: str = "runs/eqsym_surgery_A_001/data/train.csv",
    arm_b_train: str = "runs/eqsym_surgery_B_001/data/train.csv",
    seed: int = 42,
    anchor_per_family: int = 300,
) -> dict[str, Any]:
    """Assemble the Arm A and Arm B prefix-surgery training corpora.

    This callable form preserves the original per-subtype inclusion decision,
    witness selection, prefix surgery, anchor sampling, and trace verification.
    """
    train_rows = read_csv_rows(train_csv)
    solver_records = read_jsonl(solver_report)
    baseline_rows = read_csv_rows(baseline_predictions)
    train_by_id = {row["id"]: row for row in train_rows}
    solver_by_id = {row["id"]: row for row in solver_records}
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    split_train_ids, split_valid_ids = canonical_split_ids(train_rows)

    deduce_prior = operation_prior(solver_records, "deduce")
    guess_prior = operation_prior(solver_records, "guess") or GUESS_PRIOR
    deduce_rank = {op: index for index, op in enumerate(deduce_prior)}
    print(f"deduce_prior_top6: {deduce_prior[:6]}", flush=True)
    print(f"guess_prior_top4: {guess_prior[:4]}", flush=True)

    trace_dir = Path(trace_dir)
    arm_a_eq: list[dict[str, Any]] = []
    arm_b_eq: list[dict[str, Any]] = []
    trace_exclusions: list[dict[str, Any]] = []
    training_exclusions: list[dict[str, Any]] = []
    inclusion_counts: Counter[str] = Counter()
    trace_exclusion_counts: Counter[str] = Counter()
    verifier_pass_counts: Counter[str] = Counter()
    byte_continuity_counts: Counter[str] = Counter()
    byte_continuity_fixed_ids: list[str] = []

    eq_ids = [row["id"] for row in train_rows if task_variant(row.get("prompt")) == "equation_symbol_cipher"]
    for index, row_id in enumerate(eq_ids, 1):
        source = train_by_id[row_id]
        record = solver_by_id.get(row_id)
        baseline = baseline_by_id.get(row_id, {})
        if record is None:
            trace_exclusions.append({"id": row_id, "reason": "missing_solver_record", "stage": "trace_generation"})
            trace_exclusion_counts["missing_solver_record"] += 1
            continue
        puzzle = parse_puzzle(source["prompt"], row_id, source.get("answer", "").strip())
        baseline_correct = truthy(baseline.get("correct"))
        include, category, decision_meta = inclusion_decision(puzzle, record, baseline_correct, deduce_rank, guess_prior)
        if not include:
            trace_exclusions.append({"id": row_id, "reason": category, "stage": "trace_generation", **decision_meta})
            trace_exclusion_counts[category] += 1
            continue
        try:
            prefix = parse_verified_prefix(source["prompt"], baseline.get("prediction", ""), row_id, source.get("answer", ""))
            if not prefix.valid:
                raise PrefixVerificationError(prefix.cut_reason)
            prefix = add_conversion_header_byte_continuity(puzzle, prefix)
            arm_a = render_arm_a(puzzle, record, prefix, category)
            arm_b = render_arm_b(puzzle, record, category)
            undefined = undefined_letter_references(arm_a.trace, prefix)
            if undefined:
                raise TraceVerificationError("; ".join(undefined[:5]))
            verify_a = verify_rendered_trace(arm_a.trace, puzzle, prefix_info=complete_prefix_info(puzzle, prefix))
            verify_b = verify_rendered_trace(arm_b.trace, puzzle)
            arm_a_completion = trace_completion_with_prefix(prefix, arm_a.trace)
            continuity = verify_prefix_byte_continuity(prefix, arm_a_completion)
            if continuity["checked"]:
                byte_continuity_counts["checked_rows"] += 1
                byte_continuity_counts["checked_chars"] += int(continuity["chars"])
                byte_continuity_counts["checked_lines"] += int(continuity["lines"])
                byte_continuity_fixed_ids.append(row_id)
            verifier_pass_counts["arm_a"] += 1
            verifier_pass_counts["arm_b"] += 1
        except (PrefixVerificationError, TraceVerificationError, Exception) as error:
            reason = f"trace_build_{type(error).__name__}"
            trace_exclusions.append(
                {
                    "id": row_id,
                    "reason": reason,
                    "stage": "trace_generation",
                    "detail": str(error),
                    "candidate_category": category,
                }
            )
            trace_exclusion_counts[reason] += 1
            continue

        split = "valid" if row_id in split_valid_ids else "train"
        common = {
            "id": row_id,
            "answer": source.get("answer", "").strip(),
            "source": "eq_symbol",
            "split": split,
            "task_family": "equation_transformation",
            "task_variant": "equation_symbol_cipher",
            "inclusion_category": category,
            **decision_meta,
        }
        arm_a_row = {
            **common,
            "prompt": source["prompt"],
            "prefix": prefix.kept_prefix,
            "completion": arm_a_completion,
            "mode": arm_a.metadata["mode"],
            "query_op": arm_a.metadata["query_op"],
            "prefix_cut_reason": prefix.cut_reason,
            "estimated_completion_tokens": verify_a["estimated_tokens"],
            "narrowing_present": verify_a["narrowing_lines"] > 0,
            "undefined_letter_count": 0,
            **arm_a.metadata,
        }
        arm_b_row = {
            **common,
            "prompt": source["prompt"],
            "prefix": "",
            "completion": arm_b.trace,
            "mode": arm_b.metadata["mode"],
            "query_op": arm_b.metadata["query_op"],
            "estimated_completion_tokens": verify_b["estimated_tokens"],
            "narrowing_present": verify_b["narrowing_lines"] > 0,
            **arm_b.metadata,
        }
        arm_a_eq.append(arm_a_row)
        arm_b_eq.append(arm_b_row)
        inclusion_counts[category] += 1
        if index % 50 == 0:
            print(f"processed eq_symbol {index}/{len(eq_ids)} included={len(arm_a_eq)} excluded={len(trace_exclusions)}", flush=True)

    for row in arm_a_eq:
        if row["split"] == "valid":
            training_exclusions.append(
                {
                    "id": row["id"],
                    "reason": "validation_split",
                    "stage": "training_csv",
                    "inclusion_category": row["inclusion_category"],
                }
            )

    anchor_rows, anchor_counts = build_anchor_rows(
        baseline_rows,
        train_by_id,
        split_train_ids,
        seed=seed,
        per_family=anchor_per_family,
    )
    arm_a_train_eq = [row for row in arm_a_eq if row["split"] == "train"]
    arm_b_train_eq = [row for row in arm_b_eq if row["split"] == "train"]
    arm_a_train_rows = [*arm_a_train_eq, *anchor_rows]
    arm_b_train_rows = [*arm_b_train_eq, *anchor_rows]

    if [row["id"] for row in arm_a_eq] != [row["id"] for row in arm_b_eq]:
        raise RuntimeError("Arm A and Arm B generated eq-symbol row sets differ")
    if [row["id"] for row in arm_a_train_rows] != [row["id"] for row in arm_b_train_rows]:
        raise RuntimeError("Arm A and Arm B training row sets differ")
    train_csv_valid_overlap = len({row["id"] for row in arm_a_train_rows} & split_valid_ids)
    if train_csv_valid_overlap:
        raise RuntimeError(f"training CSV contains valid ids: {train_csv_valid_overlap}")
    if any(row.get("undefined_letter_count", 0) for row in arm_a_eq):
        raise RuntimeError("Arm A generated traces contain undefined letters")

    all_exclusions = [*trace_exclusions, *training_exclusions]
    write_jsonl(trace_dir / "arm_a.jsonl", arm_a_eq)
    write_jsonl(trace_dir / "arm_b.jsonl", arm_b_eq)
    write_jsonl(trace_dir / "exclusions.jsonl", all_exclusions)
    write_csv_rows(arm_a_train, arm_a_train_rows, CSV_FIELDS)
    write_csv_rows(arm_b_train, arm_b_train_rows, CSV_FIELDS)
    write_samples(trace_dir / "samples.txt", arm_a_eq, arm_b_eq)

    trace_split_counts = dict(Counter(row["split"] for row in arm_a_eq))
    combined_exclusion_counts = Counter(row["reason"] for row in all_exclusions)
    narrowing_present_count = sum(
        1 for a, b in zip(arm_a_eq, arm_b_eq) if a.get("narrowing_present") and b.get("narrowing_present")
    )
    stats = {
        "prompt_contract": "csv columns id,prompt,prefix,completion; prompt is raw Kaggle prompt; trainer chat-template appends prefix",
        "canonical_split": {
            "valid_fraction": 0.2,
            "seed": 12345,
            "train_rows": len(split_train_ids),
            "valid_rows": len(split_valid_ids),
        },
        "eq_symbol_total": len(eq_ids),
        "eq_symbol_included": len(arm_a_eq),
        "eq_symbol_excluded": len(trace_exclusions),
        "traces_generated": len(arm_a_eq),
        "traces_admitted_to_training": len(arm_a_train_eq),
        "trace_split_counts": trace_split_counts,
        "train_csv_valid_overlap": train_csv_valid_overlap,
        "inclusion_counts": dict(sorted(inclusion_counts.items())),
        "trace_exclusion_counts": dict(sorted(trace_exclusion_counts.items())),
        "training_exclusion_counts": dict(Counter(row["reason"] for row in training_exclusions)),
        "exclusion_counts": dict(sorted(combined_exclusion_counts.items())),
        "anchor_counts": anchor_counts,
        "anchor_total": len(anchor_rows),
        "arm_a_train_rows": len(arm_a_train_rows),
        "arm_b_train_rows": len(arm_b_train_rows),
        "arm_generated_row_sets_identical": [row["id"] for row in arm_a_eq] == [row["id"] for row in arm_b_eq],
        "arm_train_row_sets_identical": [row["id"] for row in arm_a_train_rows] == [row["id"] for row in arm_b_train_rows],
        "undefined_letter_traces": sum(1 for row in arm_a_eq if row.get("undefined_letter_count", 0)),
        "undefined_letter_trace_total": len(arm_a_eq),
        "byte_continuity_checked_rows": byte_continuity_counts["checked_rows"],
        "byte_continuity_checked_chars": byte_continuity_counts["checked_chars"],
        "byte_continuity_checked_lines": byte_continuity_counts["checked_lines"],
        "byte_continuity_fixed_ids": sorted(byte_continuity_fixed_ids),
        "narrowing_present_count": narrowing_present_count,
        "generated_trace_count": len(arm_a_eq),
        "trace_verifier_pass_counts": dict(verifier_pass_counts),
        "token_lengths": {
            "arm_a_eq": token_stats(arm_a_eq),
            "arm_b_eq": token_stats(arm_b_eq),
            "arm_a_train": token_stats(arm_a_train_rows),
            "arm_b_train": token_stats(arm_b_train_rows),
        },
        "deduce_prior": deduce_prior,
        "guess_prior": guess_prior,
        "training_csv_columns": CSV_FIELDS,
    }
    write_json(trace_dir / "corpus_stats.json", stats)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return stats
