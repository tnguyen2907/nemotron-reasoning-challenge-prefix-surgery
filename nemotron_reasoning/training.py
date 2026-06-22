from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from nemotron_reasoning.io_utils import read_csv_rows, write_json
from nemotron_reasoning.paths import BASE_MODEL_ID, ensure_run_layout
from nemotron_reasoning.prompts import kaggle_vllm_prompt
from nemotron_reasoning.traces import build_answer_only_sft_records

CHECKPOINT_COMPLETE_MARKER = "modal_checkpoint_complete.json"


def _distributed_context(torch: Any) -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    distributed = world_size > 1
    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def _distributed_barrier(torch: Any, context: dict[str, int | bool]) -> None:
    if context["distributed"] and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _load_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "The training stack is not installed. Run this through the Modal image "
            "or install the project with the 'modal' extra."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainerCallback": TrainerCallback,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def prompt_completion_records(records_with_metadata: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the fields TRL needs for completion-only SFT loss."""
    records: list[dict[str, str]] = []
    for record in records_with_metadata:
        prompt = record.get("prompt", "")
        completion = record.get("completion", "")
        if not prompt:
            raise ValueError(f"record {record.get('id', '<unknown>')} is missing prompt")
        if not completion:
            raise ValueError(f"record {record.get('id', '<unknown>')} is missing completion")
        records.append({"prompt": prompt, "completion": completion})
    return records


def apply_chat_template_to_records(records: list[dict[str, str]], tokenizer: Any) -> list[dict[str, str]]:
    """Format SFT prompts like the official vLLM metric prompt."""
    formatted: list[dict[str, str]] = []
    for record in records:
        out = dict(record)
        out["prompt"] = kaggle_vllm_prompt(tokenizer, record["prompt"]) + record.get("prefix", "")
        out["completion"] = record["completion"]
        formatted.append(out)
    return formatted


def _percentile(values: list[int], pct: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checkpoint_step(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


def complete_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    root = Path(checkpoint_dir)
    candidates: list[tuple[int, Path]] = []
    if not root.exists():
        return []
    for path in root.iterdir():
        step = checkpoint_step(path)
        if step is not None and (path / CHECKPOINT_COMPLETE_MARKER).is_file():
            candidates.append((step, path))
    return [path for _, path in sorted(candidates)]


def remove_incomplete_checkpoints(checkpoint_dir: str | Path) -> list[str]:
    root = Path(checkpoint_dir)
    removed: list[str] = []
    if not root.exists():
        return removed
    for path in root.iterdir():
        if checkpoint_step(path) is not None and not (path / CHECKPOINT_COMPLETE_MARKER).is_file():
            shutil.rmtree(path)
            removed.append(path.name)
    return sorted(removed)


def validate_resume_contract(
    contract_path: str | Path,
    expected: dict[str, Any],
    has_complete_checkpoint: bool,
) -> None:
    path = Path(contract_path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected and has_complete_checkpoint:
            raise ValueError(
                "Training resume contract changed while complete checkpoints exist; "
                "refusing an incompatible resume."
            )
    write_json(path, expected)


def assert_composed_lengths_within_budget(
    records: list[dict[str, str]],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    over_budget: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        text = record["prompt"] + record["completion"]
        token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
        lengths.append(token_count)
        if token_count > max_seq_length:
            over_budget.append(
                {
                    "index": index,
                    "id": record.get("id", ""),
                    "tokens": token_count,
                    "max_seq_length": max_seq_length,
                    "completion_tail": record["completion"][-120:],
                }
            )
    if over_budget:
        preview = "; ".join(f"{item['id'] or item['index']}={item['tokens']}" for item in over_budget[:10])
        raise ValueError(
            f"{len(over_budget)} composed training examples exceed max_seq_length={max_seq_length}; "
            f"refusing to train/truncate because this can cut the boxed answer. First offenders: {preview}"
        )
    return {
        "count": len(lengths),
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p99": _percentile(lengths, 0.99),
        "max": max(lengths) if lengths else 0,
        "max_seq_length": max_seq_length,
    }


def _set_training_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_trainable_parameter_regex(model: Any, regex: str) -> dict[str, int]:
    pattern = re.compile(regex)
    total = 0
    trainable = 0
    matched = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        should_train = bool(pattern.search(name))
        parameter.requires_grad = should_train
        if should_train:
            matched += 1
            trainable += parameter.numel()
    if matched == 0:
        raise ValueError(f"trainable_parameter_regex matched no parameters: {regex}")
    return {"matched_parameters": matched, "trainable_parameters": trainable, "total_parameters": total}


def _snapshot_trainable_parameters(model: Any, stack: dict[str, Any]) -> dict[str, Any]:
    torch = stack["torch"]
    snapshot: dict[str, Any] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                snapshot[name] = parameter.detach().cpu().clone()
    return snapshot


def _apply_delta_scale(model: Any, snapshot: dict[str, Any], scale: float, stack: dict[str, Any]) -> None:
    torch = stack["torch"]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            original = snapshot.get(name)
            if original is None:
                continue
            original = original.to(device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(original + (parameter - original) * scale)


def train_lora_adapter(
    run_id: str,
    train_csv: str | Path,
    base_model: str = BASE_MODEL_ID,
    limit: int | None = None,
    max_steps: int = 100,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: str = r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    warmstart_adapter: str | Path | None = None,
    trainable_parameter_regex: str | None = None,
    warmstart_delta_scale: float = 1.0,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "linear",
    warmup_steps: int = 0,
    shuffle_train_dataset: bool = True,
    chat_template_prompts: bool = False,
    gradient_checkpointing: bool = False,
    seed: int = 12345,
    checkpoint_steps: int = 56,
    checkpoint_keep: int = 2,
    checkpoint_commit_callback: Callable[[], None] | None = None,
) -> Path:
    """Train and save a PEFT LoRA adapter for the Nemotron base model."""
    stack = _load_training_dependencies()
    torch = stack["torch"]
    Dataset = stack["Dataset"]
    LoraConfig = stack["LoraConfig"]
    PeftModel = stack["PeftModel"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]
    TrainerCallback = stack["TrainerCallback"]
    SFTConfig = stack["SFTConfig"]
    SFTTrainer = stack["SFTTrainer"]
    distributed_context = _distributed_context(torch)
    is_main_process = bool(distributed_context["is_main"])
    _set_training_seed(seed, torch)

    if checkpoint_steps <= 0:
        raise ValueError("checkpoint_steps must be positive")
    if checkpoint_keep <= 0:
        raise ValueError("checkpoint_keep must be positive")

    root = ensure_run_layout(run_id)
    output_dir = root / "checkpoints"
    removed_incomplete = remove_incomplete_checkpoints(output_dir) if is_main_process else []
    config = {
        "run_id": run_id,
        "base_model": base_model,
        "train_csv": str(train_csv),
        "limit": limit,
        "max_steps": max_steps,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "warmstart_adapter": str(warmstart_adapter) if warmstart_adapter else None,
        "trainable_parameter_regex": trainable_parameter_regex,
        "warmstart_delta_scale": warmstart_delta_scale,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_steps": warmup_steps,
        "shuffle_train_dataset": shuffle_train_dataset,
        "chat_template_prompts": chat_template_prompts,
        "gradient_checkpointing": gradient_checkpointing,
        "seed": seed,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_keep": checkpoint_keep,
        "loss_scope": "completion_only",
        "sft_input_contract": "prompt/completion; answer-only completion is derived when completion is absent",
    }
    resume_contract = {
        "version": 1,
        "run_id": run_id,
        "train_csv_sha256": sha256_file(train_csv),
        **config,
    }
    if is_main_process:
        checkpoints = complete_checkpoints(output_dir)
        validate_resume_contract(
            root / "config" / "resume_contract.json",
            resume_contract,
            has_complete_checkpoint=bool(checkpoints),
        )
        write_json(root / "config" / "train_lora_adapter.json", config)
    _distributed_barrier(torch, distributed_context)
    checkpoints = complete_checkpoints(output_dir)

    all_rows = read_csv_rows(train_csv)
    rows = all_rows if limit in (None, 0) else all_rows[:limit]
    sft_records = build_answer_only_sft_records(rows)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if chat_template_prompts:
        sft_records = apply_chat_template_to_records(sft_records, tokenizer)
    token_length_summary = assert_composed_lengths_within_budget(sft_records, tokenizer, max_seq_length)
    if is_main_process:
        write_json(root / "eval" / "train_token_lengths.json", token_length_summary)
    dataset = Dataset.from_list(prompt_completion_records(sft_records))

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    device_map = {"": int(distributed_context["local_rank"])} if distributed_context["distributed"] else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = None
    trainable_summary = None
    warmstart_snapshot = None
    peft_weight_conversion = None

    def _ready_adapter_converter(
        model: Any,
        peft_config: Any,
        adapter_state_dict: dict[str, Any],
        adapter_name: str = "default",
        **_: Any,
    ) -> dict[str, Any]:
        converted: dict[str, Any] = {}
        for key, value in adapter_state_dict.items():
            renamed = key.replace("base_model.model.model.", "base_model.model.backbone.")
            renamed = renamed.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head.")
            converted[renamed] = value
        return converted

    if warmstart_adapter:
        warmstart_path = Path(warmstart_adapter)
        if not (warmstart_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Warm-start adapter is missing adapter_config.json: {warmstart_path}")
        try:
            import peft.utils.transformers_weight_conversion as peft_weight_conversion
        except ImportError:
            peft_weight_conversion = None

        if peft_weight_conversion is None:
            model = PeftModel.from_pretrained(model, warmstart_path, is_trainable=True)
        else:
            original_converter = peft_weight_conversion.convert_peft_adapter_state_dict_for_transformers

            try:
                # The public Kien/Tinker adapter is already in PEFT/vLLM-ready layout.
                # PEFT 0.19 + Transformers 5 currently attempts a Nemotron-H
                # conversion path that fails before loading. The needed mapping
                # for this adapter is only the model.model -> backbone rename.
                peft_weight_conversion.convert_peft_adapter_state_dict_for_transformers = _ready_adapter_converter
                model = PeftModel.from_pretrained(model, warmstart_path, is_trainable=True)
            finally:
                peft_weight_conversion.convert_peft_adapter_state_dict_for_transformers = original_converter
        if trainable_parameter_regex:
            trainable_summary = _set_trainable_parameter_regex(model, trainable_parameter_regex)
        if warmstart_delta_scale != 1.0:
            warmstart_snapshot = _snapshot_trainable_parameters(model, stack)
    else:
        if trainable_parameter_regex:
            raise ValueError("--trainable-parameter-regex requires --warmstart-run-id")
        if lora_rank > 32:
            raise ValueError("Kaggle requires LoRA rank <= 32")
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_strategy="no",
        save_total_limit=checkpoint_keep,
        bf16=True,
        packing=False,
        max_length=max_seq_length,
        completion_only_loss=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    class StepCheckpointCallback(TrainerCallback):  # type: ignore[misc, valid-type]
        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if state.global_step > 0 and state.global_step < max_steps and state.global_step % checkpoint_steps == 0:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if not state.is_world_process_zero:
                return control
            checkpoint = output_dir / f"checkpoint-{state.global_step}"
            write_json(
                checkpoint / CHECKPOINT_COMPLETE_MARKER,
                {
                    "global_step": state.global_step,
                    "saved_at_unix": time.time(),
                    "resume_contract_sha256": hashlib.sha256(
                        json.dumps(resume_contract, sort_keys=True).encode("utf-8")
                    ).hexdigest().upper(),
                },
            )
            if checkpoint_commit_callback is not None:
                checkpoint_commit_callback()
            print(f"Committed complete checkpoint at step {state.global_step}: {checkpoint}", flush=True)
            return control

    trainer_kwargs["callbacks"] = [StepCheckpointCallback()]
    trainer_cls = SFTTrainer
    if not shuffle_train_dataset:
        from torch.utils.data import SequentialSampler

        class SequentialSFTTrainer(SFTTrainer):  # type: ignore[misc, valid-type]
            def _get_train_sampler(self, train_dataset=None):  # type: ignore[no-untyped-def]
                return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

        trainer_cls = SequentialSFTTrainer
    trainer = trainer_cls(**trainer_kwargs)
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint is not None:
        print(f"Resuming from complete checkpoint: {resume_checkpoint}", flush=True)
    if removed_incomplete:
        print(f"Removed incomplete checkpoints: {removed_incomplete}", flush=True)
    if resume_checkpoint is not None and peft_weight_conversion is not None and warmstart_adapter:
        original_converter = peft_weight_conversion.convert_peft_adapter_state_dict_for_transformers
        try:
            # Trainer checkpoint resume calls PeftModel.load_adapter again.
            # Keep the same Nemotron adapter-key shim active for that reload.
            peft_weight_conversion.convert_peft_adapter_state_dict_for_transformers = _ready_adapter_converter
            train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        finally:
            peft_weight_conversion.convert_peft_adapter_state_dict_for_transformers = original_converter
    else:
        train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    if warmstart_snapshot is not None:
        _apply_delta_scale(trainer.model, warmstart_snapshot, warmstart_delta_scale, stack)

    adapter_dir = root / "adapter"
    if is_main_process:
        trainer.save_model(str(adapter_dir))
        (adapter_dir / "README.md").write_text(
            "# Nemotron Reasoning Challenge LoRA Adapter\n\n"
            "PEFT LoRA adapter generated for the NVIDIA Nemotron Model Reasoning Challenge.\n",
            encoding="utf-8",
        )
        tokenizer.save_pretrained(adapter_dir / "tokenizer")
        write_json(
            root / "eval" / "train_summary.json",
            {
                "trained_rows": len(rows),
                "sft_record_count": len(sft_records),
                "max_steps": max_steps,
                "loss_scope": "completion_only",
                "trainable_summary": trainable_summary,
                "warmstart_delta_scale": warmstart_delta_scale,
                "chat_template_prompts": chat_template_prompts,
                "gradient_checkpointing": gradient_checkpointing,
                "seed": seed,
                "checkpoint_steps": checkpoint_steps,
                "checkpoint_keep": checkpoint_keep,
                "distributed": bool(distributed_context["distributed"]),
                "world_size": int(distributed_context["world_size"]),
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": (
                    per_device_train_batch_size
                    * gradient_accumulation_steps
                    * int(distributed_context["world_size"])
                ),
                "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                "removed_incomplete_checkpoints": removed_incomplete,
                "train_metrics": train_result.metrics,
                "token_length_summary": token_length_summary,
            },
        )
    _distributed_barrier(torch, distributed_context)
    return adapter_dir
