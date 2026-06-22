from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from nemotron_reasoning.io_utils import read_csv_rows, write_csv_rows, write_json
from nemotron_reasoning.metric import score_prediction
from nemotron_reasoning.paths import BASE_MODEL_ID, ensure_run_layout
from nemotron_reasoning.prompts import kaggle_vllm_prompt
from nemotron_reasoning.task_types import task_family, task_variant


def _summarize_prediction_groups(rows: list[dict[str, str]], group_key: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get(group_key, "unknown") or "unknown", []).append(row)

    summaries: list[dict[str, object]] = []
    for group, group_rows in sorted(groups.items()):
        scored_rows = [row for row in group_rows if row.get("correct") in {"True", "False"}]
        correct = sum(row.get("correct") == "True" for row in scored_rows)
        generated_tokens = [int(row.get("generated_tokens") or 0) for row in group_rows]
        summaries.append(
            {
                group_key: group,
                "row_count": len(group_rows),
                "scored_count": len(scored_rows),
                "correct": correct,
                "accuracy": correct / len(scored_rows) if scored_rows else 0.0,
                "method_counts": dict(Counter(row.get("method") or "unknown" for row in group_rows)),
                "finish_reason_counts": dict(
                    Counter(row.get("finish_reason") or "unknown" for row in group_rows)
                ),
                "max_generated_tokens": max(generated_tokens, default=0),
                "mean_generated_tokens": (
                    sum(generated_tokens) / len(generated_tokens) if generated_tokens else 0.0
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (float(item["accuracy"]), -int(item["row_count"]), str(item[group_key])),
    )


def run_vllm_inference(
    run_id: str,
    eval_csv: str | Path,
    base_model: str = BASE_MODEL_ID,
    limit: int = 8,
    start: int = 0,
    max_tokens: int = 7680,
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 64,
    output_name: str = "eval_predictions.csv",
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = True,
    use_adapter: bool = True,
) -> dict[str, Any]:
    """Run deterministic vLLM inference using the competition metric settings."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Run this through the Modal image.") from exc

    root = ensure_run_layout(run_id)
    adapter_dir = root / "adapter"
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit < 0:
        raise ValueError("limit must be non-negative; use 0 for all rows after start")
    output_file = Path(output_name).name
    if output_file != output_name:
        raise ValueError(f"output_name must be a filename, got {output_name!r}")

    all_rows = [row for row in read_csv_rows(eval_csv) if row.get("prompt")]
    rows = all_rows[start : None if limit == 0 else start + limit]
    if use_adapter and not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter is missing adapter_config.json: {adapter_dir}")

    llm_kwargs: dict[str, Any] = {
        "model": base_model,
        "trust_remote_code": True,
        "enable_lora": use_adapter,
        "max_model_len": 8192,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": "auto",
        "enable_prefix_caching": enable_prefix_caching,
        "enable_chunked_prefill": enable_chunked_prefill,
    }
    if use_adapter:
        llm_kwargs["max_lora_rank"] = 32
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    tokenizer = llm.get_tokenizer()
    prompts = [kaggle_vllm_prompt(tokenizer, row["prompt"]) for row in rows]
    if use_adapter:
        outputs = llm.generate(
            prompts,
            sampling_params=sampling,
            lora_request=LoRARequest("adapter", 1, str(adapter_dir)),
        )
    else:
        outputs = llm.generate(prompts, sampling_params=sampling)

    fieldnames = [
        "id",
        "task_family",
        "task_variant",
        "answer",
        "prediction",
        "output_tail",
        "extracted",
        "correct",
        "method",
        "finish_reason",
        "generated_tokens",
        "prompt_tokens",
    ]
    prediction_rows: list[dict[str, str]] = []
    finish_reasons: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    generated_token_counts: list[int] = []
    correct = 0
    scored_count = 0
    for row, output in zip(rows, outputs, strict=True):
        generation = output.outputs[0] if output.outputs else None
        text = generation.text if generation else ""
        finish_reason = str(getattr(generation, "finish_reason", "") or "")
        generated_tokens = len(getattr(generation, "token_ids", []) or []) if generation else 0
        prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
        finish_reasons[finish_reason or "unknown"] += 1
        generated_token_counts.append(generated_tokens)

        answer = row.get("answer", "")
        if answer:
            scored = score_prediction(text, answer)
            correct += int(scored.correct)
            scored_count += 1
            extracted = scored.extracted
            is_correct = str(scored.correct)
            method = scored.method
            methods[method] += 1
        else:
            extracted = ""
            is_correct = ""
            method = ""
        prediction_rows.append(
            {
                "id": row["id"],
                "task_family": task_family(row.get("prompt")),
                "task_variant": task_variant(row.get("prompt")),
                "answer": answer,
                "prediction": text,
                "output_tail": text[-500:],
                "extracted": extracted,
                "correct": is_correct,
                "method": method,
                "finish_reason": finish_reason,
                "generated_tokens": str(generated_tokens),
                "prompt_tokens": str(prompt_tokens),
            }
        )

    pred_path = root / "predictions" / output_file
    write_csv_rows(pred_path, prediction_rows, fieldnames)
    summary = {
        "row_count": len(rows),
        "scored_count": scored_count,
        "correct": correct,
        "accuracy": correct / scored_count if scored_count else 0.0,
        "predictions": str(pred_path),
        "eval_csv": str(eval_csv),
        "start": start,
        "limit": limit,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "temperature": 0.0,
        "top_p": 1.0,
        "enable_prefix_caching": enable_prefix_caching,
        "enable_chunked_prefill": enable_chunked_prefill,
        "use_adapter": use_adapter,
        "prompt_template": "official_metric_chat_template_enable_thinking",
        "whole_list_generate": True,
        "method_counts": dict(methods),
        "finish_reason_counts": dict(finish_reasons),
        "max_generated_tokens": max(generated_token_counts, default=0),
        "mean_generated_tokens": (
            sum(generated_token_counts) / len(generated_token_counts) if generated_token_counts else 0.0
        ),
        "by_task_family": _summarize_prediction_groups(prediction_rows, "task_family"),
        "by_task_variant": _summarize_prediction_groups(prediction_rows, "task_variant"),
    }
    write_json(root / "eval" / f"{Path(output_file).stem}_summary.json", summary)
    write_json(root / "eval" / "eval_summary.json", summary)
    return summary
