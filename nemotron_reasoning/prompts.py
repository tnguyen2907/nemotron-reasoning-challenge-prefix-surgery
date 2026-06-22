from __future__ import annotations

from typing import Any

KAGGLE_BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def kaggle_user_content(prompt: str) -> str:
    """Return the user prompt text used by the official Kaggle metric."""
    return f"{prompt}{KAGGLE_BOXED_INSTRUCTION}"


def kaggle_vllm_prompt(tokenizer: Any, prompt: str) -> str:
    """Format a prompt like the official metric, falling back to plain text."""
    user_content = kaggle_user_content(prompt)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception:
        return user_content


def boxed_answer_completion(answer: str) -> str:
    """Default answer-only SFT target for rows that do not have completions."""
    return f"\n\\boxed{{{answer.strip()}}}"
