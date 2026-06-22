from __future__ import annotations

import math
import re
from dataclasses import dataclass

BOXED_RE = re.compile(r"\\boxed\{")
BINARY_RE = re.compile(r"^[01]+$")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

@dataclass(frozen=True)
class ScoreResult:
    prediction: str
    target: str
    extracted: str
    correct: bool
    method: str


def _strip_math_space(value: str) -> str:
    return value.strip()


def extract_boxed_answer(text: str | None) -> str | None:
    """Extract boxed content using the official metric notebook strategy."""
    if text is None:
        return None

    boxed_starts = list(BOXED_RE.finditer(text))
    if not boxed_starts:
        return None

    matches: list[str] = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[index + 1].start() if index + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)

    non_empty = [match.strip() for match in matches if match.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip()


def fallback_extract_answer(text: str | None) -> str:
    """Fallback answer extraction from the pulled Kaggle metric notebook."""
    if text is None:
        return "NOT_FOUND"

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def extract_answer(text: str | None) -> tuple[str, str]:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed, "boxed"
    return fallback_extract_answer(text), "fallback"


def answers_match(predicted: str, target: str, rel_tol: float = 1e-2, abs_tol: float = 1e-5) -> bool:
    pred = _strip_math_space(predicted)
    gold = _strip_math_space(target)
    if BINARY_RE.fullmatch(gold):
        return pred.lower() == gold.lower()
    try:
        gold_float = float(gold)
        pred_float = float(pred)
        return math.isclose(pred_float, gold_float, rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return pred.lower() == gold.lower()


def score_prediction(prediction: str, target: str, rel_tol: float = 1e-2) -> ScoreResult:
    extracted, method = extract_answer(prediction)
    return ScoreResult(
        prediction=prediction,
        target=target,
        extracted=extracted,
        correct=answers_match(extracted, target, rel_tol=rel_tol),
        method=method,
    )


def score_rows(rows: list[dict[str, str]], prediction_key: str = "prediction", target_key: str = "answer") -> dict[str, object]:
    results = [score_prediction(row.get(prediction_key, ""), row.get(target_key, "")) for row in rows]
    correct = sum(result.correct for result in results)
    return {
        "row_count": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "boxed_count": sum(result.method == "boxed" for result in results),
        "fallback_count": sum(result.method == "fallback" for result in results),
    }
