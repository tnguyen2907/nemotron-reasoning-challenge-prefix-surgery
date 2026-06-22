from __future__ import annotations

from nemotron_reasoning.prompts import boxed_answer_completion, kaggle_user_content
from nemotron_reasoning.task_types import task_family


def make_answer_only_sft_record(row: dict[str, str]) -> dict[str, str]:
    """Create a minimal supervised example when no completion is supplied."""
    prompt = row["prompt"].strip()
    answer = row["answer"].strip()
    prompt_text = kaggle_user_content(prompt)
    completion = boxed_answer_completion(answer)
    return {
        "id": row["id"],
        "task_family": task_family(prompt),
        "prompt": prompt_text,
        "completion": completion,
        "text": f"{prompt_text}{completion}",
        "answer": answer,
        "trace_type": "answer_only",
    }


def build_answer_only_sft_records(
    rows: list[dict[str, str]],
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Preserve explicit trace completions or derive a boxed answer target."""
    records: list[dict[str, str]] = []
    for row in rows:
        if row.get("completion"):
            prompt = row.get("prompt", "").strip()
            completion = row.get("completion", "")
            if not prompt:
                raise ValueError(f"row {row.get('id', '<unknown>')} is missing prompt")
            record = {"prompt": prompt, "completion": completion}
            if "id" in row:
                record["id"] = row.get("id", "")
            if "prefix" in row:
                record["prefix"] = row.get("prefix", "")
            records.append(record)
        else:
            records.append(make_answer_only_sft_record(row))
        if limit is not None and len(records) >= limit:
            break
    return records
