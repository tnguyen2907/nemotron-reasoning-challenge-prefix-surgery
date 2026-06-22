"""Parser and gold-conditioned solver for equation_symbol_cipher puzzles.

Puzzle structure (confirmed over all 823 train rows on June 11, 2026):
every example LHS and every query is exactly 5 glyphs, ``ABoCD``. The
operator glyph is at index 2; operands are two encoded digits each.

The solver uses the lkevincc public symbolic-solver operation universe as
the source-backed first pass, but uses a local pattern-enumeration + join
search instead of permutation backtracking. Active solving is base-10 only:
An exhaustive data scan found that all 823 training rows have <=10 content symbols.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Callable

QUERY_RE = re.compile(r"result for:\s*(.+?)\s*$", re.MULTILINE)
BASE = 10
DIGIT_COUNT = 10


class SearchTimeout(Exception):
    """Raised when a per-row search budget is exhausted."""


_DEADLINE: float | None = None


def _set_deadline(seconds: float | None) -> None:
    global _DEADLINE
    _DEADLINE = None if seconds is None else time.monotonic() + seconds


def _check_deadline() -> None:
    if _DEADLINE is not None and time.monotonic() > _DEADLINE:
        raise SearchTimeout()


@dataclass(frozen=True)
class Equation:
    left: str
    op: str
    right: str
    result: str


@dataclass
class Puzzle:
    puzzle_id: str
    examples: list[Equation]
    query: Equation
    gold: str | None = None

    @property
    def example_ops(self) -> set[str]:
        return {eq.op for eq in self.examples}

    @property
    def subtype(self) -> str:
        return "deduce" if self.query.op in self.example_ops else "guess"

    @property
    def digit_glyphs(self) -> list[str]:
        # Port of lkevincc's answer-hint orphan-symbol rule: symbols that
        # appear only in the gold answer must still be available for mapping.
        ops = self.example_ops | {self.query.op}
        seen: dict[str, None] = {}
        for eq in [*self.examples, self.query]:
            for ch in eq.left + eq.right + eq.result:
                if ch not in ops:
                    seen.setdefault(ch, None)
        if self.gold:
            for ch in self.gold:
                if ch.strip() and ch not in ops:
                    seen.setdefault(ch, None)
        return list(seen)


class PuzzleParseError(ValueError):
    pass


def _split_expression(text: str) -> tuple[str, str, str]:
    if len(text) != 5:
        raise PuzzleParseError(f"expression {text!r} is not 5 glyphs")
    return text[:2], text[2], text[3:]


def parse_puzzle(prompt: str, puzzle_id: str = "", gold: str | None = None) -> Puzzle:
    examples: list[Equation] = []
    for line in prompt.strip().splitlines():
        line = line.strip()
        if " = " in line and not line.lower().startswith(("in alice", "now,")):
            lhs, rhs = line.split(" = ", 1)
            left, op, right = _split_expression(lhs.strip())
            examples.append(Equation(left, op, right, rhs.strip()))
    match = QUERY_RE.search(prompt)
    if not examples or match is None:
        raise PuzzleParseError(f"could not parse puzzle {puzzle_id!r}")
    left, op, right = _split_expression(match.group(1).strip())
    gold_value = gold.strip() if isinstance(gold, str) else None
    return Puzzle(puzzle_id, examples, Equation(left, op, right, ""), gold_value)


# --- Operation registry -------------------------------------------------
# Provenance: lkevincc0/kaggle-nemotron-equation-symbolic
# src/solver_eq_symbolic.py and rust/alice_sovler_helper/src/lib.rs.


def _ss(a: int, b: int) -> int | None:
    return a - b if a >= b else None


def _rs(a: int, b: int) -> int | None:
    return b - a if b >= a else None


def _fd(a: int, b: int) -> int | None:
    return a // b if b else None


def _rd(a: int, b: int) -> int | None:
    return b // a if a else None


def _mo(a: int, b: int) -> int | None:
    return a % b if b else None


def _rm(a: int, b: int) -> int | None:
    return b % a if a else None


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b) if a and b else 0


def _nz(v: int | None) -> int | None:
    return v if v is not None and v >= 0 else None


def _off(fn: Callable[[int, int], int | None], delta: int) -> Callable[[int, int], int | None]:
    def inner(a: int, b: int) -> int | None:
        value = fn(a, b)
        if value is None:
            return None
        return _nz(value + delta)

    return inner


NUMERIC_OPERATIONS: dict[str, Callable[[int, int], int | None]] = {
    "add": lambda a, b: a + b,
    "sub": _ss,
    "rsub": _rs,
    "absdiff": lambda a, b: abs(a - b),
    "neg_absdiff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
    "gcd": gcd,
    "lcm": _lcm,
    "fdiv": _fd,
    "rdiv": _rd,
    "mod": _mo,
    "rmod": _rm,
    "min": min,
    "max": max,
    "add_m1": _off(lambda a, b: a + b, -1),
    "add_p1": _off(lambda a, b: a + b, 1),
    "add_m2": _off(lambda a, b: a + b, -2),
    "add_p2": _off(lambda a, b: a + b, 2),
    "mul_m1": _off(lambda a, b: a * b, -1),
    "mul_p1": _off(lambda a, b: a * b, 1),
    "mul_m2": _off(lambda a, b: a * b, -2),
    "mul_p2": _off(lambda a, b: a * b, 2),
    "absdiff_m1": _off(lambda a, b: abs(a - b), -1),
    "absdiff_p1": _off(lambda a, b: abs(a - b), 1),
    "absdiff_m2": _off(lambda a, b: abs(a - b), -2),
    "absdiff_p2": _off(lambda a, b: abs(a - b), 2),
    "sub_m1": _off(_ss, -1),
    "sub_p1": _off(_ss, 1),
    "rsub_m1": _off(_rs, -1),
    "rsub_p1": _off(_rs, 1),
    "mul_half": lambda a, b: (a * b) // 2,
    "mul_double": lambda a, b: a * b * 2,
    "sq_diff": lambda a, b: (a - b) ** 2,
    "sq_sum": lambda a, b: (a + b) ** 2,
    "mul_plus_a": lambda a, b: a * b + a,
    "mul_plus_b": lambda a, b: a * b + b,
    "mul_minus_a": lambda a, b: _nz(a * b - a),
    "mul_minus_b": lambda a, b: _nz(a * b - b),
    "a2_plus_b": lambda a, b: a * a + b,
    "a_plus_b2": lambda a, b: a + b * b,
    "xor": lambda a, b: a ^ b,
    "band": lambda a, b: a & b,
    "bor": lambda a, b: a | b,
    "sub_signed": lambda a, b: a - b,
    "rsub_signed": lambda a, b: b - a,
}

SPECIAL_OPS = {"concat_fwd", "concat_rev"}
SIGNED_OPS = {"sub_signed", "rsub_signed"}
OP_ALIASES = {
    "concat": "concat_fwd",
    "rconcat": "concat_rev",
    "add_plus1": "add_p1",
    "add_minus1": "add_m1",
    "mul_plus1": "mul_p1",
    "mul_minus1": "mul_m1",
    "floordiv": "fdiv",
}


def canonical_op_name(name: str) -> str:
    return OP_ALIASES.get(name, name)


def _value_op_for_compat(name: str) -> Callable[[str, str], str | None]:
    canonical = canonical_op_name(name)

    def inner(da: str, db: str) -> str | None:
        if canonical == "concat_fwd":
            return da + db
        if canonical == "concat_rev":
            return db + da
        value = NUMERIC_OPERATIONS[canonical](int(da), int(db))
        if value is None or value < 0:
            return None
        return str(value)

    return inner


OPERATION_ORDER = [
    "add",
    "sub",
    "rsub",
    "absdiff",
    "neg_absdiff",
    "mul",
    "gcd",
    "lcm",
    "fdiv",
    "rdiv",
    "mod",
    "rmod",
    "min",
    "max",
    "add_m1",
    "add_p1",
    "add_m2",
    "add_p2",
    "mul_m1",
    "mul_p1",
    "mul_m2",
    "mul_p2",
    "absdiff_m1",
    "absdiff_p1",
    "absdiff_m2",
    "absdiff_p2",
    "sub_m1",
    "sub_p1",
    "rsub_m1",
    "rsub_p1",
    "mul_half",
    "mul_double",
    "sq_diff",
    "sq_sum",
    "mul_plus_a",
    "mul_plus_b",
    "mul_minus_a",
    "mul_minus_b",
    "a2_plus_b",
    "a_plus_b2",
    "xor",
    "band",
    "bor",
    "sub_signed",
    "rsub_signed",
    "concat_fwd",
    "concat_rev",
]

OPERATIONS = {name: _value_op_for_compat(name) for name in OPERATION_ORDER}
for alias in OP_ALIASES:
    OPERATIONS[alias] = _value_op_for_compat(alias)

OP_PRIORITY = {
    "*": [
        "mul",
        "mul_m1",
        "mul_p1",
        "mul_m2",
        "mul_p2",
        "absdiff",
        "add",
        "gcd",
        "lcm",
        "mul_half",
        "mul_double",
        "mul_plus_a",
        "mul_plus_b",
        "mul_minus_a",
        "mul_minus_b",
        "sq_diff",
        "sq_sum",
        "concat_fwd",
        "concat_rev",
    ],
    "+": [
        "add",
        "add_m1",
        "add_p1",
        "add_m2",
        "add_p2",
        "mul",
        "absdiff",
        "gcd",
        "lcm",
        "sq_sum",
        "mul_plus_a",
        "mul_plus_b",
        "concat_fwd",
        "concat_rev",
    ],
    "-": [
        "rsub",
        "absdiff",
        "sub",
        "sub_signed",
        "rsub_signed",
        "absdiff_m1",
        "absdiff_p1",
        "absdiff_m2",
        "absdiff_p2",
        "sub_m1",
        "sub_p1",
        "rsub_m1",
        "rsub_p1",
        "mul",
        "add",
        "neg_absdiff",
        "gcd",
        "lcm",
        "sq_diff",
        "concat_fwd",
        "concat_rev",
    ],
    "/": ["fdiv", "rdiv", "mul", "add", "absdiff", "concat_fwd", "concat_rev"],
}

DEFAULT_PRIORITY = [
    "add",
    "absdiff",
    "mul",
    "sub",
    "rsub",
    "sub_signed",
    "rsub_signed",
    "add_m1",
    "add_p1",
    "mul_m1",
    "mul_p1",
    "absdiff_m1",
    "absdiff_p1",
    "gcd",
    "lcm",
    "concat_fwd",
    "concat_rev",
    "sub_m1",
    "sub_p1",
    "rsub_m1",
    "rsub_p1",
    "add_m2",
    "add_p2",
    "mul_m2",
    "mul_p2",
    "absdiff_m2",
    "absdiff_p2",
    "mul_half",
    "mul_double",
    "sq_diff",
    "sq_sum",
    "mul_plus_a",
    "mul_plus_b",
    "mul_minus_a",
    "mul_minus_b",
    "a2_plus_b",
    "a_plus_b2",
    "neg_absdiff",
    "fdiv",
    "rdiv",
    "mod",
    "rmod",
    "min",
    "max",
    "xor",
    "band",
    "bor",
]

TIER0 = {
    "add",
    "sub",
    "rsub",
    "sub_signed",
    "rsub_signed",
    "absdiff",
    "neg_absdiff",
    "mul",
    "gcd",
    "lcm",
    "concat_fwd",
    "concat_rev",
}
TIER1 = TIER0 | {"fdiv", "rdiv", "mod", "rmod", "min", "max"}
TIER2 = TIER1 | {
    "add_m1",
    "add_p1",
    "mul_m1",
    "mul_p1",
    "absdiff_m1",
    "absdiff_p1",
    "sub_m1",
    "sub_p1",
    "rsub_m1",
    "rsub_p1",
}
TIER3 = set(OPERATION_ORDER)
TIER_ORDER = [TIER0, TIER1, TIER2, TIER3]


@dataclass(frozen=True)
class Mode:
    reverse_operands: bool = False
    reverse_result: bool = False
    zfill_result: bool = False
    name: str = ""
    reverse_digits: bool = False

    def describe(self) -> str:
        if self.name:
            return self.name
        flags = []
        if self.reverse_operands:
            flags.append("rev_operands")
        if self.reverse_result:
            flags.append("rev_result")
        if self.zfill_result:
            flags.append("zfill")
        return "+".join(flags) if flags else "plain"

    @property
    def semantic_key(self) -> str:
        return "reverse_digits" if self.reverse_digits or self.reverse_operands else "standard"


MODES = [
    Mode(name="standard"),
    Mode(name="little_endian", reverse_digits=True),
    Mode(name="alice", reverse_digits=True),
]


@dataclass
class Solution:
    mapping: dict[str, int]
    op_assignment: dict[str, str]
    mode: Mode

    def encode(self, value_digits: str) -> str | None:
        inverse = {digit: glyph for glyph, digit in self.mapping.items()}
        out = []
        for ch in value_digits:
            glyph = inverse.get(int(ch))
            if glyph is None:
                return None
            out.append(glyph)
        return "".join(out)


@dataclass(frozen=True)
class PartialMapping:
    items: tuple[tuple[str, int], ...]
    used_mask: int

    @property
    def mapping(self) -> dict[str, int]:
        return dict(self.items)


def _make_partial(mapping: dict[str, int]) -> PartialMapping:
    return PartialMapping(tuple(sorted(mapping.items())), sum(1 << d for d in mapping.values()))


@dataclass(frozen=True)
class UnconditionedResult:
    answers: set[str]
    status: str
    solution_limit_reached: bool = False


def _merge_partials(left: PartialMapping, right: PartialMapping) -> PartialMapping | None:
    merged = dict(left.items)
    used_mask = left.used_mask
    for glyph, digit in right.items:
        existing = merged.get(glyph)
        if existing is not None:
            if existing != digit:
                return None
            continue
        if used_mask & (1 << digit):
            return None
        merged[glyph] = digit
        used_mask |= 1 << digit
    return PartialMapping(tuple(sorted(merged.items())), used_mask)


def pattern_key(text: str) -> tuple[int, ...]:
    """Canonical first-occurrence pattern for a glyph/digit string."""
    seen: dict[str, int] = {}
    out: list[int] = []
    for ch in text:
        if ch not in seen:
            seen[ch] = len(seen)
        out.append(seen[ch])
    return tuple(out)


def _digits_match_pattern(digits: tuple[int, ...], pattern: tuple[int, ...]) -> bool:
    by_pattern: dict[int, int] = {}
    used_digits: dict[int, int] = {}
    for digit, pat in zip(digits, pattern):
        if pat in by_pattern:
            if by_pattern[pat] != digit:
                return False
        elif digit in used_digits:
            return False
        else:
            by_pattern[pat] = digit
            used_digits[digit] = pat
    return True


def _int_to_base_digits(value: int, width: int | None = None) -> list[int]:
    if value == 0:
        digits = [0]
    else:
        digits = []
        while value:
            digits.append(value % BASE)
            value //= BASE
        digits.reverse()
    if width is not None and len(digits) < width:
        digits = [0] * (width - len(digits)) + digits
    return digits


def _result_body_and_sign(eq: Equation) -> tuple[str, bool]:
    has_sign = len(eq.result) > 1 and eq.result[0] == eq.op
    return (eq.result[1:], True) if has_sign else (eq.result, False)


def _concat_result(eq: Equation, op_name: str, overlap: bool = False) -> str | None:
    if op_name == "concat_fwd":
        if overlap and eq.left[-1] == eq.right[0]:
            return eq.left + eq.right[1:]
        return eq.left + eq.right
    if op_name == "concat_rev":
        if overlap and eq.right[-1] == eq.left[0]:
            return eq.right + eq.left[1:]
        return eq.right + eq.left
    return None


def _concat_result_options(eq: Equation, op_name: str) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for overlap, mode_name in ((False, "pure_concat"), (True, "pure_concat_overlap")):
        value = _concat_result(eq, op_name, overlap=overlap)
        if value is not None and value not in {seen for seen, _ in options}:
            options.append((value, mode_name))
    return options


def _pure_concat_solutions(
    puzzle: Puzzle,
    gold_conditioned: bool,
    max_solutions: int,
) -> list[Solution]:
    if puzzle.query.op not in puzzle.example_ops:
        return []
    query_examples = [eq for eq in puzzle.examples if eq.op == puzzle.query.op]
    out: list[Solution] = []
    for op_name in ("concat_fwd", "concat_rev"):
        if not all(eq.result in {value for value, _ in _concat_result_options(eq, op_name)} for eq in query_examples):
            continue
        for predicted, mode_name in _concat_result_options(puzzle.query, op_name):
            if gold_conditioned and puzzle.gold is not None and predicted != puzzle.gold:
                continue
            out.append(Solution({}, {puzzle.query.op: op_name}, Mode(name=mode_name)))
            if len(out) >= max_solutions:
                break
        if len(out) >= max_solutions:
            break
    return out


def _equation_usable(eq: Equation, op_chars: set[str]) -> bool:
    body, _ = _result_body_and_sign(eq)
    return all(ch not in op_chars for ch in eq.left + eq.right + body)


def _priority(op_char: str) -> list[str]:
    preferred = OP_PRIORITY.get(op_char, DEFAULT_PRIORITY)
    out = [name for name in preferred if name in TIER3]
    seen = set(out)
    out.extend(name for name in OPERATION_ORDER if name not in seen)
    return out


def _op_candidates_for_group(op_char: str, eqs: list[Equation]) -> list[str]:
    signs = {_result_body_and_sign(eq)[1] for eq in eqs}
    priority = _priority(op_char)
    if signs == {True}:
        candidates = [name for name in priority if name in {"neg_absdiff", "sub_signed", "rsub_signed"}]
    elif signs == {False}:
        candidates = [name for name in priority if name != "neg_absdiff"]
    else:
        candidates = [name for name in priority if name in {"sub_signed", "rsub_signed"}]
    if all((not _result_body_and_sign(eq)[1] and len(_result_body_and_sign(eq)[0]) == 4) for eq in eqs):
        for name in ("concat_fwd", "concat_rev"):
            if name not in candidates:
                candidates.append(name)
    return candidates


def _tiered_candidates(names: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for tier in TIER_ORDER:
        for name in names:
            if name in tier and name not in seen:
                ordered.append(name)
                seen.add(name)
    return ordered


@lru_cache(maxsize=None)
def _candidate_digit_sequences(
    op_name: str,
    semantic_mode: str,
    result_len: int,
    has_sign: bool,
    pattern: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    reverse_digits = semantic_mode == "reverse_digits"
    for left in range(100):
        l0, l1 = divmod(left, 10)
        for right in range(100):
            r0, r1 = divmod(right, 10)
            left_value = l1 * 10 + l0 if reverse_digits else left
            right_value = r1 * 10 + r0 if reverse_digits else right
            if op_name == "concat_fwd":
                if has_sign or result_len != 4:
                    continue
                result_digits = [l0, l1, r0, r1]
            elif op_name == "concat_rev":
                if has_sign or result_len != 4:
                    continue
                result_digits = [r0, r1, l0, l1]
            else:
                raw = NUMERIC_OPERATIONS[op_name](left_value, right_value)
                if raw is None:
                    continue
                if op_name in SIGNED_OPS:
                    if (raw < 0) != has_sign:
                        continue
                    magnitude = abs(raw)
                elif op_name == "neg_absdiff":
                    if not has_sign or raw < 0:
                        continue
                    magnitude = raw
                else:
                    if has_sign or raw < 0:
                        continue
                    magnitude = raw
                if magnitude >= BASE**result_len:
                    continue
                result_digits = _int_to_base_digits(magnitude, result_len)
                if reverse_digits:
                    result_digits = result_digits[::-1]
            digits = (l0, l1, r0, r1, *result_digits)
            if _digits_match_pattern(digits, pattern):
                out.append(digits)
    return tuple(out)


def _equation_candidates(eq: Equation, op_name: str, mode: Mode) -> list[PartialMapping]:
    body, has_sign = _result_body_and_sign(eq)
    chars = eq.left + eq.right + body
    pattern = pattern_key(chars)
    sequences = _candidate_digit_sequences(
        op_name,
        mode.semantic_key,
        len(body),
        has_sign,
        pattern,
    )
    candidates: list[PartialMapping] = []
    for digits in sequences:
        mapping: dict[str, int] = {}
        ok = True
        for glyph, digit in zip(chars, digits):
            existing = mapping.get(glyph)
            if existing is not None and existing != digit:
                ok = False
                break
            mapping[glyph] = digit
        if ok:
            candidates.append(_make_partial(mapping))
    return candidates


def _partial_key(partial: PartialMapping, glyphs: tuple[str, ...]) -> tuple[int | None, ...]:
    mapping = dict(partial.items)
    return tuple(mapping.get(glyph) for glyph in glyphs)


def _join_partial_lists(lists: list[list[PartialMapping]]) -> list[PartialMapping]:
    if not lists:
        return []
    joined = lists[0]
    for next_list in lists[1:]:
        _check_deadline()
        left_glyphs = set().union(*(dict(p.items).keys() for p in joined)) if joined else set()
        right_glyphs = set().union(*(dict(p.items).keys() for p in next_list)) if next_list else set()
        shared = tuple(sorted(left_glyphs & right_glyphs))
        merged: list[PartialMapping] = []
        if shared:
            index: dict[tuple[int | None, ...], list[PartialMapping]] = defaultdict(list)
            for right in next_list:
                index[_partial_key(right, shared)].append(right)
            for left in joined:
                for right in index.get(_partial_key(left, shared), []):
                    combo = _merge_partials(left, right)
                    if combo is not None:
                        merged.append(combo)
        else:
            for left in joined:
                for right in next_list:
                    combo = _merge_partials(left, right)
                    if combo is not None:
                        merged.append(combo)
        joined = merged
        if not joined:
            return []
    return joined


def _join_group_for_op(eqs: list[Equation], op_name: str, mode: Mode) -> list[PartialMapping]:
    candidate_lists = [_equation_candidates(eq, op_name, mode) for eq in eqs]
    if any(not candidates for candidates in candidate_lists):
        return []
    candidate_lists.sort(key=len)
    return _join_partial_lists(candidate_lists)


def _solve_with_join(
    puzzle: Puzzle,
    equations: list[Equation],
    modes: list[Mode],
    max_solutions: int,
) -> list[Solution]:
    found: list[Solution] = []
    for mode in modes:
        _check_deadline()
        by_op: dict[str, list[Equation]] = defaultdict(list)
        for eq in equations:
            by_op[eq.op].append(eq)

        group_options: list[tuple[str, list[tuple[str, list[PartialMapping]]]]] = []
        for op_char, op_eqs in by_op.items():
            options: list[tuple[str, list[PartialMapping]]] = []
            for op_name in _tiered_candidates(_op_candidates_for_group(op_char, op_eqs)):
                _check_deadline()
                joined = _join_group_for_op(op_eqs, op_name, mode)
                if joined:
                    options.append((op_name, joined))
            if not options:
                break
            options.sort(key=lambda item: (len(item[1]), _priority(op_char).index(item[0]) if item[0] in _priority(op_char) else 999))
            group_options.append((op_char, options))
        else:
            group_options.sort(key=lambda item: min(len(candidates) for _, candidates in item[1]))

            def recurse(
                index: int,
                mapping: PartialMapping,
                op_assignment: dict[str, str],
            ) -> None:
                _check_deadline()
                if len(found) >= max_solutions:
                    return
                if index == len(group_options):
                    found.append(Solution(dict(mapping.items), dict(op_assignment), mode))
                    return
                op_char, options = group_options[index]
                for op_name, candidates in options:
                    op_assignment[op_char] = op_name
                    for candidate in candidates:
                        merged = _merge_partials(mapping, candidate)
                        if merged is not None:
                            recurse(index + 1, merged, op_assignment)
                            if len(found) >= max_solutions:
                                return
                    del op_assignment[op_char]

            recurse(0, PartialMapping((), 0), {})
        if len(found) >= max_solutions:
            return found
    return found


def _apply_operation_to_query(
    puzzle: Puzzle,
    solution: Solution,
    expected_len: int | None = None,
) -> tuple[str | None, int | None]:
    eq = puzzle.query
    op_name = canonical_op_name(solution.op_assignment.get(eq.op, ""))
    if op_name in SPECIAL_OPS:
        answer = _concat_result(eq, op_name, overlap=solution.mode.name == "pure_concat_overlap")
        if expected_len is not None and answer is not None and len(answer) != expected_len:
            return None, None
        return answer, None

    mapping = solution.mapping
    if any(ch not in mapping for ch in eq.left + eq.right):
        return None, None
    reverse = solution.mode.semantic_key == "reverse_digits"
    l0, l1 = mapping[eq.left[0]], mapping[eq.left[1]]
    r0, r1 = mapping[eq.right[0]], mapping[eq.right[1]]
    left_value = (l1 * 10 + l0) if reverse else (l0 * 10 + l1)
    right_value = (r1 * 10 + r0) if reverse else (r0 * 10 + r1)
    inverse = {digit: glyph for glyph, digit in mapping.items()}

    if op_name not in NUMERIC_OPERATIONS:
        return None, None
    raw = NUMERIC_OPERATIONS[op_name](left_value, right_value)
    if raw is None:
        return None, None
    if op_name in SIGNED_OPS:
        prefix = eq.op if raw < 0 else ""
        magnitude = abs(raw)
        numeric = raw
    elif op_name == "neg_absdiff":
        if raw < 0:
            return None, None
        prefix = eq.op
        magnitude = raw
        numeric = -raw
    else:
        if raw < 0:
            return None, None
        prefix = ""
        magnitude = raw
        numeric = raw
    digits = _int_to_base_digits(magnitude, expected_len)
    if expected_len is not None and len(digits) > expected_len:
        return None, None
    if reverse:
        digits = digits[::-1]
    try:
        encoded = prefix + "".join(inverse[digit] for digit in digits)
    except KeyError:
        return None, None
    return encoded, numeric


def _legacy_predict(puzzle: Puzzle, solution: Solution, expected_len: int | None = None) -> str | None:
    eq = puzzle.query
    left = "".join(str(solution.mapping[c]) for c in eq.left)
    right = "".join(str(solution.mapping[c]) for c in eq.right)
    if solution.mode.reverse_operands:
        left, right = left[::-1], right[::-1]
    op_name = canonical_op_name(solution.op_assignment.get(eq.op, ""))
    if op_name == "concat_fwd":
        produced = left + right
    elif op_name == "concat_rev":
        produced = right + left
    else:
        fn = NUMERIC_OPERATIONS.get(op_name)
        if fn is None:
            return None
        value = fn(int(left), int(right))
        if value is None or value < 0:
            return None
        produced = str(value)
    if expected_len is not None:
        if solution.mode.zfill_result and len(produced) < expected_len:
            produced = produced.zfill(expected_len)
        if len(produced) != expected_len:
            return None
    if solution.mode.reverse_result:
        produced = produced[::-1]
    return solution.encode(produced)


def predict(puzzle: Puzzle, solution: Solution, expected_len: int | None = None) -> str | None:
    if not solution.mode.name:
        return _legacy_predict(puzzle, solution, expected_len)
    answer, _ = _apply_operation_to_query(puzzle, solution, expected_len)
    return answer


def solve_puzzle(
    puzzle: Puzzle,
    gold_conditioned: bool = True,
    max_solutions: int = 1,
    modes: list[Mode] | None = None,
) -> list[Solution]:
    pure_concat = _pure_concat_solutions(puzzle, gold_conditioned, max_solutions)
    if pure_concat:
        return pure_concat
    op_chars = puzzle.example_ops | {puzzle.query.op}
    equations = [eq for eq in puzzle.examples if _equation_usable(eq, op_chars)]
    if gold_conditioned and puzzle.gold:
        query_equation = Equation(puzzle.query.left, puzzle.query.op, puzzle.query.right, puzzle.gold)
        if not _equation_usable(query_equation, op_chars):
            return []
        equations.append(query_equation)
    return _solve_with_join(puzzle, equations, modes or MODES, max_solutions)


def _query_body_widths(puzzle: Puzzle) -> list[int]:
    op_chars = puzzle.example_ops | {puzzle.query.op}
    widths: set[int] = set()
    for eq in puzzle.examples:
        if eq.op == puzzle.query.op and _equation_usable(eq, op_chars):
            body, _ = _result_body_and_sign(eq)
            if body:
                widths.add(len(body))
    if puzzle.gold:
        body, _ = _result_body_and_sign(
            Equation(puzzle.query.left, puzzle.query.op, puzzle.query.right, puzzle.gold)
        )
        if body:
            widths.add(len(body))
    return sorted(widths or {1, 2, 3, 4})


def _encode_digit_options(
    prefix: str,
    digits: list[int],
    mapping: dict[str, int],
    content_glyphs: list[str],
    limit: int,
) -> set[str]:
    inverse = {digit: glyph for glyph, digit in mapping.items()}
    needed_digits: list[int] = []
    for digit in digits:
        if digit not in inverse and digit not in needed_digits:
            needed_digits.append(digit)
    available_glyphs = [glyph for glyph in content_glyphs if glyph not in inverse.values()]
    answers: set[str] = set()
    if len(needed_digits) > len(available_glyphs):
        return answers
    for glyphs in itertools.permutations(available_glyphs, len(needed_digits)):
        extended_inverse = dict(inverse)
        extended_inverse.update(zip(needed_digits, glyphs))
        answers.add(prefix + "".join(extended_inverse[digit] for digit in digits))
        if len(answers) >= limit:
            break
    return answers


def _possible_query_answers(puzzle: Puzzle, solution: Solution, limit: int) -> set[str]:
    op_name = canonical_op_name(solution.op_assignment.get(puzzle.query.op, ""))
    if op_name in SPECIAL_OPS:
        answer = _concat_result(
            puzzle.query,
            op_name,
            overlap=solution.mode.name == "pure_concat_overlap",
        )
        return {answer} if answer is not None else set()
    if op_name not in NUMERIC_OPERATIONS:
        return set()

    base_mapping = dict(solution.mapping)
    query_glyphs = list(dict.fromkeys(puzzle.query.left + puzzle.query.right))
    missing_query_glyphs = [glyph for glyph in query_glyphs if glyph not in base_mapping]
    used_digits = set(base_mapping.values())
    available_digits = [digit for digit in range(DIGIT_COUNT) if digit not in used_digits]
    if len(missing_query_glyphs) > len(available_digits):
        return set()

    content_glyphs = puzzle.digit_glyphs
    answers: set[str] = set()
    widths = _query_body_widths(puzzle)
    reverse = solution.mode.semantic_key == "reverse_digits"
    for digits_for_missing in itertools.permutations(available_digits, len(missing_query_glyphs)):
        mapping = dict(base_mapping)
        mapping.update(zip(missing_query_glyphs, digits_for_missing))
        if any(ch not in mapping for ch in puzzle.query.left + puzzle.query.right):
            continue
        l0, l1 = mapping[puzzle.query.left[0]], mapping[puzzle.query.left[1]]
        r0, r1 = mapping[puzzle.query.right[0]], mapping[puzzle.query.right[1]]
        left_value = (l1 * 10 + l0) if reverse else (l0 * 10 + l1)
        right_value = (r1 * 10 + r0) if reverse else (r0 * 10 + r1)
        raw = NUMERIC_OPERATIONS[op_name](left_value, right_value)
        if raw is None:
            continue
        if op_name in SIGNED_OPS:
            prefix = puzzle.query.op if raw < 0 else ""
            magnitude = abs(raw)
        elif op_name == "neg_absdiff":
            if raw < 0:
                continue
            prefix = puzzle.query.op
            magnitude = raw
        else:
            if raw < 0:
                continue
            prefix = ""
            magnitude = raw
        for width in widths:
            if magnitude >= BASE**width:
                continue
            result_digits = _int_to_base_digits(magnitude, width)
            if reverse:
                result_digits = result_digits[::-1]
            answers.update(
                _encode_digit_options(
                    prefix,
                    result_digits,
                    mapping,
                    content_glyphs,
                    limit - len(answers),
                )
            )
            if len(answers) >= limit:
                return answers
    return answers


def unconditioned_answer_result(
    puzzle: Puzzle,
    limit: int = 2,
    solution_limit: int = 128,
) -> UnconditionedResult:
    answers: set[str] = set()
    if puzzle.query.op not in puzzle.example_ops:
        return UnconditionedResult(answers, "unknown")
    for solution in _pure_concat_solutions(puzzle, gold_conditioned=False, max_solutions=limit):
        answers.update(_possible_query_answers(puzzle, solution, limit - len(answers)))
        if len(answers) >= limit:
            return UnconditionedResult(answers, "ambiguous")
    if answers:
        return UnconditionedResult(answers, "unique")
    op_chars = puzzle.example_ops | {puzzle.query.op}
    equations = [eq for eq in puzzle.examples if _equation_usable(eq, op_chars)]
    solutions = _solve_with_join(puzzle, equations, MODES, max_solutions=solution_limit)
    for solution in solutions:
        answers.update(_possible_query_answers(puzzle, solution, limit - len(answers)))
        if len(answers) >= limit:
            return UnconditionedResult(answers, "ambiguous")
    if len(solutions) >= solution_limit:
        return UnconditionedResult(answers, "unknown", solution_limit_reached=True)
    if len(answers) == 1:
        return UnconditionedResult(answers, "unique")
    if len(answers) >= limit:
        return UnconditionedResult(answers, "ambiguous")
    return UnconditionedResult(answers, "unknown")


def unconditioned_answers(puzzle: Puzzle, limit: int = 2) -> set[str]:
    return unconditioned_answer_result(puzzle, limit=limit).answers


# --- Batch report -------------------------------------------------------


def _solve_row(args: tuple) -> dict:
    puzzle_id, prompt, gold, solve_budget, *rest = args
    uniqueness_budget = rest[0] if rest else solve_budget
    started = time.monotonic()
    record: dict = {"id": puzzle_id, "gold": gold}

    def finish() -> dict:
        _set_deadline(None)
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return record

    try:
        puzzle = parse_puzzle(prompt, puzzle_id, gold)
    except PuzzleParseError as error:
        record.update({"status": "parse_error", "error": str(error)})
        return finish()
    record["subtype"] = puzzle.subtype
    _set_deadline(solve_budget)
    try:
        solutions = solve_puzzle(puzzle, gold_conditioned=True, max_solutions=1)
    except SearchTimeout:
        record["status"] = "timeout"
        return finish()
    except Exception as error:
        record.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return finish()
    if not solutions:
        record["status"] = "unsolved"
        return finish()
    solution = solutions[0]
    gold_body = gold[1:] if gold and len(gold) > 1 and gold[0] == puzzle.query.op else gold
    predicted = predict(puzzle, solution, expected_len=len(gold_body) if gold_body else None)
    record.update(
        {
            "status": "solved",
            "predicted": predicted,
            "matches_gold": predicted == gold,
            "ops": {glyph: name for glyph, name in solution.op_assignment.items()},
            "query_op": puzzle.query.op,
            "mode": solution.mode.describe(),
            "mapping": solution.mapping,
        }
    )
    if puzzle.subtype == "deduce":
        if uniqueness_budget <= 0:
            record["unique"] = None
            record["uniqueness_skipped"] = True
        else:
            _set_deadline(uniqueness_budget)
            try:
                uniqueness = unconditioned_answer_result(puzzle)
                record["unconditioned_answers"] = sorted(uniqueness.answers)
                record["uniqueness_status"] = uniqueness.status
                if uniqueness.solution_limit_reached:
                    record["uniqueness_solution_limit_reached"] = True
                if uniqueness.status == "unique":
                    record["unique"] = True
                elif uniqueness.status == "ambiguous":
                    record["unique"] = False
                else:
                    record["unique"] = None
            except SearchTimeout:
                record["unique"] = None
                record["uniqueness_status"] = "unknown"
                record["uniqueness_timeout"] = True
            except Exception as error:
                record["unique"] = None
                record["uniqueness_status"] = "unknown"
                record["uniqueness_error_type"] = type(error).__name__
                record["uniqueness_error"] = str(error)
    return finish()


def run_report(
    train_csv: Path,
    out_jsonl: Path,
    limit: int = 0,
    workers: int = 8,
    row_budget: float = 20.0,
    uniqueness_budget: float = 5.0,
    progress_every: int = 25,
) -> dict:
    import csv

    from nemotron_reasoning.task_types import task_variant

    rows: list[tuple[str, str, str, float]] = []
    with train_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if task_variant(row.get("prompt")) == "equation_symbol_cipher":
                rows.append(
                    (
                        row["id"],
                        row["prompt"],
                        (row.get("answer") or "").strip(),
                        row_budget,
                        uniqueness_budget,
                    )
                )
    if limit:
        rows = rows[:limit]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with out_jsonl.open("w", encoding="utf-8") as handle:
        if workers <= 1:
            for index, args in enumerate(rows, 1):
                record = _solve_row(args)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if progress_every > 0 and (index % progress_every == 0 or index == len(rows)):
                    print(f"progress: {index}/{len(rows)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_solve_row, args): args[0] for args in rows}
                for index, future in enumerate(as_completed(futures), 1):
                    puzzle_id = futures[future]
                    try:
                        record = future.result()
                    except Exception as error:
                        record = {
                            "id": puzzle_id,
                            "status": "worker_error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    if progress_every > 0 and (index % progress_every == 0 or index == len(rows)):
                        print(f"progress: {index}/{len(rows)}", flush=True)

    summary: dict = {"total": len(records)}
    for subtype in ("deduce", "guess"):
        subset = [r for r in records if r.get("subtype") == subtype]
        solved = [r for r in subset if r.get("status") == "solved"]
        summary[subtype] = {
            "total": len(subset),
            "solved": len(solved),
            "solved_match_gold": sum(1 for r in solved if r.get("matches_gold")),
            "unique": sum(1 for r in solved if r.get("unique")),
            "ambiguous": sum(1 for r in solved if r.get("unique") is False),
            "timeouts": sum(1 for r in subset if r.get("status") == "timeout"),
            "unsolved": sum(1 for r in subset if r.get("status") == "unsolved"),
            "errors": sum(1 for r in subset if r.get("status") in {"error", "worker_error"}),
            "uniqueness_unknown": sum(
                1
                for r in solved
                if r.get("unique") is None and r.get("uniqueness_status") == "unknown"
            ),
        }
    summary["parse_errors"] = sum(1 for r in records if r.get("status") == "parse_error")
    summary["timeouts"] = sum(1 for r in records if r.get("status") == "timeout")
    summary["uniqueness_timeouts"] = sum(1 for r in records if r.get("uniqueness_timeout"))
    summary["worker_errors"] = sum(1 for r in records if r.get("status") == "worker_error")
    summary["errors"] = sum(1 for r in records if r.get("status") == "error")
    op_counter: dict[str, int] = {}
    op_counter_by_subtype: dict[str, dict[str, int]] = {}
    for record in records:
        if record.get("status") == "solved":
            name = record["ops"].get(record.get("query_op", ""), "?")
            op_counter[name] = op_counter.get(name, 0) + 1
            subtype = record.get("subtype", "?")
            op_counter_by_subtype.setdefault(subtype, {})
            op_counter_by_subtype[subtype][name] = op_counter_by_subtype[subtype].get(name, 0) + 1
    summary["query_op_frequency"] = dict(sorted(op_counter.items(), key=lambda kv: -kv[1]))
    summary["query_op_frequency_by_subtype"] = {
        subtype: dict(sorted(counter.items(), key=lambda kv: -kv[1]))
        for subtype, counter in sorted(op_counter_by_subtype.items())
    }
    return summary
