from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from nemotron_reasoning.symbol_cipher import (
    BASE,
    NUMERIC_OPERATIONS,
    SIGNED_OPS,
    SPECIAL_OPS,
    Equation,
    MODES,
    Mode,
    Puzzle,
    Solution,
    _apply_operation_to_query,
    _concat_result,
    _equation_candidates,
    _equation_usable,
    _int_to_base_digits,
    _possible_query_answers,
    _pure_concat_solutions,
    _result_body_and_sign,
    _solve_with_join,
    canonical_op_name,
    parse_puzzle,
)

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
CHECK_RE = re.compile(
    r"^(?:(?:Check E(?P<index>\d+)|Query)|(?:Narrow E(?P<narrow_index>\d+) (?P<narrow_kind>fit|reject|crosscheck))):\s+"
    r"(?:candidates=(?P<candidates>\d+);\s+)?op=(?P<op>[a-z0-9_]+);\s+"
    r"left=(?P<left>-?\d+);\s+right=(?P<right>-?\d+);\s+raw=(?P<raw>-?\d+);\s+"
    r"encoded=(?P<encoded>.*?);\s+expected=(?P<expected>.*?)\.\s*$"
)
NATURAL_CHECK_RE = re.compile(
    r"^(?P<prefix>Check E(?P<index>\d+)|Query|Narrow E(?P<narrow_index>\d+) (?P<narrow_kind>fit|reject|crosscheck)):\s+"
    r"(?:trying (?P<try_phrase>.*?):\s+)?"
    r"(?P<left_letters>[A-Za-z]+)\s+(?P<op_label>[A-Za-z])\s+(?P<right_letters>[A-Za-z]+)\s+means\s+"
    r"(?P<left>-?\d+)\s+and\s+(?P<right>-?\d+);\s+"
    r"(?P<calc>.*?);\s+encoded as\s+(?P<encoded>.*?),\s+"
    r"(?P<relation>matching|not matching)\s+(?P<expected>.*?)\.\s*$"
)
NATURAL_CONCAT_RE = re.compile(
    r"^(?P<prefix>Check E(?P<index>\d+)|Query|Narrow E(?P<narrow_index>\d+) (?P<narrow_kind>fit|reject|crosscheck)):\s+"
    r"(?P<phrase>concatenation|reverse concatenation)\((?P<left_letters>[A-Za-z]+),\s+(?P<right_letters>[A-Za-z]+)\)\s+=\s+"
    r"(?P<first>[A-Za-z]+)\s+\|\|\s+(?P<second>[A-Za-z]+)\s+=\s+(?P<encoded>[A-Za-z]+),\s+"
    r"(?P<relation>matching|not matching)\s+(?P<expected>[A-Za-z]+)\.\s*$"
)
TABLE_RE = re.compile(r"^\s*(?P<glyph>.+?)\s*->\s*(?P<label>[A-Za-z])\s*$")
INLINE_PAIR_RE = re.compile(r"(?P<glyph>\S)\s*->\s*(?P<label>[A-Za-z])(?=$|,|\s*->|\s*:)")
LABEL_INTRO_RE = re.compile(r"^Label the examples E1-E(?P<count>\d+) in order\.$")
ARM_A_BANNED_SNIPPETS = ("Correcting", "op=")

DEDUCE_PRIOR_TOP4 = ["mul", "add", "sub_signed", "concat_fwd"]
GUESS_PRIOR = ["add", "sub_signed", "mul", "concat_fwd"]


@dataclass(frozen=True)
class ConvertedLine:
    example_index: int
    kind: str
    expected: str
    observed: str
    line: str
    ok: bool


@dataclass(frozen=True)
class ByteContinuityCheck:
    name: str
    original_start: int
    original_end: int
    completion_start: int = 0


@dataclass(frozen=True)
class PrefixInfo:
    valid: bool
    cut_offset: int
    kept_prefix: str
    symbol_to_letter: dict[str, str]
    operator_to_letter: dict[str, str]
    converted_lines: list[ConvertedLine] = field(default_factory=list)
    cut_reason: str = ""
    next_expected_line: str = ""
    original_prediction: str = ""
    byte_continuity_end: int | None = None
    byte_continuity_checks: tuple[ByteContinuityCheck, ...] = ()

    @property
    def letter_to_symbol(self) -> dict[str, str]:
        return {letter: glyph for glyph, letter in self.symbol_to_letter.items()}


@dataclass(frozen=True)
class TraceBuild:
    trace: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WitnessProgram:
    answer: str
    query_op: str
    mode: str


class PrefixVerificationError(ValueError):
    pass


class TraceVerificationError(ValueError):
    pass


def _line_spans(text: str) -> Iterable[tuple[int, int, str]]:
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        yield start, end, line
        start = end


def _clean_bracket_text(value: str) -> str:
    value = value.strip().rstrip(":").strip()
    bracket_pairs = [("「", "」"), ("\u300c", "\u300d"), ("ã€Œ", "ã€")]
    for left, right in bracket_pairs:
        if value.startswith(left) and value.endswith(right):
            return value[len(left) : -len(right)]
    return value


def _letters_only(value: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", value))


def _fresh_label(used: set[str]) -> str:
    for code in range(ord("A"), ord("Z") + 1):
        label = chr(code)
        if label not in used:
            return label
    for code in range(ord("a"), ord("z") + 1):
        label = chr(code)
        if label not in used:
            return label
    raise PrefixVerificationError("ran out of fresh labels")


def _fresh_operator_label(used: set[str]) -> str:
    for label in ["x", "y", "z", "u", "v", "w", "p", "q", "r", "s", "t"]:
        if label not in used:
            return label
    return _fresh_label(used)


def _letter_expr(eq: Equation, symbol_to_letter: dict[str, str], operator_to_letter: dict[str, str]) -> str:
    return (
        "".join(symbol_to_letter[ch] for ch in eq.left)
        + operator_to_letter[eq.op]
        + "".join(symbol_to_letter[ch] for ch in eq.right)
    )


def _letter_expr_display(eq: Equation, symbol_to_letter: dict[str, str], operator_to_letter: dict[str, str]) -> str:
    return (
        "".join(symbol_to_letter[ch] for ch in eq.left)
        + f" {operator_to_letter[eq.op]} "
        + "".join(symbol_to_letter[ch] for ch in eq.right)
    )


def _letter_result(result: str, symbol_to_letter: dict[str, str], operator_to_letter: dict[str, str]) -> str:
    out: list[str] = []
    for ch in result:
        if ch in symbol_to_letter:
            out.append(symbol_to_letter[ch])
        elif ch in operator_to_letter:
            out.append(operator_to_letter[ch])
        else:
            raise PrefixVerificationError(f"missing letter for glyph {ch!r}")
    return "".join(out)


def parse_verified_prefix(prompt: str, prediction: str, row_id: str = "", gold: str | None = None) -> PrefixInfo:
    """Verify Kien-style letter-table and conversion prefix, returning the cut point."""
    puzzle = parse_puzzle(prompt, row_id, gold)
    symbol_to_letter: dict[str, str] = {}
    operator_to_letter: dict[str, str] = {}
    converted: list[ConvertedLine] = []
    in_operator_section = False
    in_conversion = False
    current_example = -1
    last_good_end = 0
    cut_reason = "empty prediction"
    next_expected = ""

    def fail(start: int, reason: str, expected_line: str = "", valid: bool = False) -> PrefixInfo:
        return PrefixInfo(
            valid=valid,
            cut_offset=last_good_end if last_good_end else start,
            kept_prefix=prediction[: last_good_end if last_good_end else start],
            symbol_to_letter=dict(symbol_to_letter),
            operator_to_letter=dict(operator_to_letter),
            converted_lines=list(converted),
            cut_reason=reason,
            next_expected_line=expected_line,
            original_prediction=prediction,
            byte_continuity_end=last_good_end if last_good_end else start,
        )

    def assign_label(glyph: str, label: str, force_operator: bool = False) -> PrefixInfo | None:
        target = operator_to_letter if force_operator or glyph in (puzzle.example_ops | {puzzle.query.op}) else symbol_to_letter
        other = symbol_to_letter if target is operator_to_letter else operator_to_letter
        if glyph in target:
            return None if target[glyph] == label else fail(0, f"conflicting label for {glyph!r}", valid=True)
        if glyph in other:
            return None
        used_by_symbol = {value: key for key, value in symbol_to_letter.items()}
        used_by_operator = {value: key for key, value in operator_to_letter.items()}
        if label in used_by_symbol and used_by_symbol[label] != glyph:
            return fail(0, f"duplicate symbol label {label!r}", valid=True)
        if label in used_by_operator and used_by_operator[label] != glyph:
            return fail(0, f"duplicate operator label {label!r}", valid=True)
        target[glyph] = label
        return None

    def learn_inline_pairs(line: str) -> PrefixInfo | None:
        for match in INLINE_PAIR_RE.finditer(line):
            problem = assign_label(match.group("glyph"), match.group("label"))
            if problem is not None:
                return problem
        return None

    for start, end, raw_line in _line_spans(prediction):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped:
            last_good_end = end
            continue
        if "assign letters" in lower or lower.startswith("first,"):
            last_good_end = end
            continue
        if stripped == "Operators:":
            in_operator_section = True
            last_good_end = end
            continue
        if stripped == "Operands:":
            in_operator_section = False
            last_good_end = end
            continue
        if "->" in stripped and ":" in stripped and not in_conversion:
            candidate = stripped.split(":", 1)[1].strip()
            match = TABLE_RE.match(candidate)
            if match:
                problem = assign_label(match.group("glyph").strip(), match.group("label"), force_operator="operator" in lower)
                if problem is not None:
                    return problem
                last_good_end = end
                continue
        if "converting examples" in lower:
            in_conversion = True
            in_operator_section = False
            last_good_end = end
            continue
        if not in_conversion:
            match = TABLE_RE.match(line)
            if match:
                glyph = match.group("glyph").strip()
                label = match.group("label")
                problem = assign_label(glyph, label, force_operator=in_operator_section)
                if problem is not None:
                    return problem
                last_good_end = end
                continue
            # Keep harmless prose before the conversion block only if the table
            # has not started yet. Once table parsing starts, unknown text is a cut.
            if not symbol_to_letter and not operator_to_letter:
                last_good_end = end
                continue
            return fail(start, "unparseable table line", valid=bool(symbol_to_letter or operator_to_letter))

        learn_problem = learn_inline_pairs(line)
        if learn_problem is not None:
            return learn_problem
        example_digit_glyphs: list[str] = []
        for eq in puzzle.examples:
            for glyph in eq.left + eq.right + eq.result:
                if glyph not in (puzzle.example_ops | {puzzle.query.op}) and glyph not in example_digit_glyphs:
                    example_digit_glyphs.append(glyph)
        missing_symbols = [glyph for glyph in example_digit_glyphs if glyph not in symbol_to_letter]
        missing_ops = [glyph for glyph in puzzle.example_ops if glyph not in operator_to_letter]
        if missing_symbols or missing_ops:
            return fail(start, f"missing table entries symbols={missing_symbols} ops={missing_ops}", valid=True)

        if (stripped.startswith("「") or stripped.startswith("ã€Œ")) and " = " in stripped:
            current_example += 1
            if current_example >= len(puzzle.examples):
                return fail(start, "too many example headers", valid=True)
            try:
                lhs_part, rhs_part = stripped.split(" = ", 1)
                lhs = _clean_bracket_text(lhs_part)
                rhs = _clean_bracket_text(rhs_part)
            except ValueError:
                return fail(start, "unparseable example header", valid=True)
            eq = puzzle.examples[current_example]
            if lhs != eq.left + eq.op + eq.right or rhs != eq.result:
                expected_header = f"「{eq.left}{eq.op}{eq.right}」 = 「{eq.result}」"
                return fail(start, "example header does not match prompt", expected_header, valid=True)
            last_good_end = end
            continue

        if "input:" in lower or "output:" in lower:
            if current_example < 0 or current_example >= len(puzzle.examples):
                return fail(start, "conversion line before example header", valid=True)
            eq = puzzle.examples[current_example]
            kind = "input" if "input:" in lower else "output"
            try:
                expected = _letter_expr(eq, symbol_to_letter, operator_to_letter) if kind == "input" else _letter_result(eq.result, symbol_to_letter, operator_to_letter)
            except (KeyError, PrefixVerificationError) as error:
                return fail(start, f"unverifiable {kind} conversion on E{current_example + 1}: {error}", valid=True)
            observed = _letters_only(line.rsplit("->", 1)[-1]) if "->" in line else ""
            ok = observed == expected
            converted.append(ConvertedLine(current_example, kind, expected, observed, line, ok))
            if not ok:
                return fail(start, f"bad {kind} conversion on E{current_example + 1}", expected, valid=True)
            last_good_end = end
            continue

        # Invented sections or operator conclusions are beyond the verified
        # prefix. Cut before them rather than trusting a heuristic derailment.
        return PrefixInfo(
            valid=True,
            cut_offset=last_good_end,
            kept_prefix=prediction[:last_good_end],
            symbol_to_letter=dict(symbol_to_letter),
            operator_to_letter=dict(operator_to_letter),
            converted_lines=list(converted),
            cut_reason=f"cut before unverified line: {stripped[:80]}",
            next_expected_line="",
            original_prediction=prediction,
            byte_continuity_end=last_good_end,
        )

    return PrefixInfo(
        valid=True,
        cut_offset=last_good_end,
        kept_prefix=prediction[:last_good_end],
        symbol_to_letter=dict(symbol_to_letter),
        operator_to_letter=dict(operator_to_letter),
        converted_lines=list(converted),
        cut_reason="end of prediction",
        next_expected_line="",
        original_prediction=prediction,
        byte_continuity_end=last_good_end,
    )


def mode_from_record(record: dict[str, Any]) -> Mode:
    name = record.get("mode") or "standard"
    if name in {"little_endian", "alice"}:
        return Mode(name=name, reverse_digits=True)
    return Mode(name=name)


def solution_from_record(record: dict[str, Any]) -> Solution:
    return Solution(
        mapping={str(k): int(v) for k, v in (record.get("mapping") or {}).items()},
        op_assignment={str(k): str(v) for k, v in (record.get("ops") or {}).items()},
        mode=mode_from_record(record),
    )


def complete_prefix_info(puzzle: Puzzle, info: PrefixInfo) -> PrefixInfo:
    completed, _ = complete_prefix_info_with_definitions(puzzle, info)
    return completed


def complete_prefix_info_with_definitions(puzzle: Puzzle, info: PrefixInfo) -> tuple[PrefixInfo, list[str]]:
    symbol_to_letter = dict(info.symbol_to_letter)
    operator_to_letter = dict(info.operator_to_letter)
    used = set(symbol_to_letter.values()) | set(operator_to_letter.values())
    added: list[str] = []
    definition_lines: list[str] = []
    for glyph in puzzle.digit_glyphs:
        if glyph not in symbol_to_letter:
            label = _fresh_label(used)
            symbol_to_letter[glyph] = label
            used.add(label)
            added.append(f"{glyph}->{label}")
            definition_lines.append(f"{glyph} -> {label}")
    for glyph in sorted(puzzle.example_ops | {puzzle.query.op}):
        if glyph not in operator_to_letter:
            label = _fresh_operator_label(used)
            operator_to_letter[glyph] = label
            used.add(label)
            added.append(f"{glyph}->{label}")
            definition_lines.append(f"{glyph} -> {label}")
    reason = info.cut_reason
    if added:
        reason = f"{reason}; completed missing labels {', '.join(added)}"
    return (
        PrefixInfo(
            valid=info.valid,
            cut_offset=info.cut_offset,
            kept_prefix=info.kept_prefix,
            symbol_to_letter=symbol_to_letter,
            operator_to_letter=operator_to_letter,
            converted_lines=info.converted_lines,
            cut_reason=reason,
            next_expected_line=info.next_expected_line,
            original_prediction=info.original_prediction,
            byte_continuity_end=info.byte_continuity_end,
            byte_continuity_checks=info.byte_continuity_checks,
        ),
        definition_lines,
    )


def _body_width_for_result(result: str, op: str) -> int:
    body, _ = _result_body_and_sign(Equation("", op, "", result))
    return len(body)


def _pseudo_puzzle(eq: Equation) -> Puzzle:
    return Puzzle("trace", [], Equation(eq.left, eq.op, eq.right, ""), None)


def _pair_value(pair: str, mapping: dict[str, int], mode: Mode) -> tuple[str, int]:
    digits = "".join(str(mapping[ch]) for ch in pair)
    if mode.semantic_key == "reverse_digits":
        return digits, int(digits[::-1])
    return digits, int(digits)


def _encoded_letters(encoded: str, info: PrefixInfo) -> str:
    out: list[str] = []
    for ch in encoded:
        if ch in info.symbol_to_letter:
            out.append(info.symbol_to_letter[ch])
        elif ch in info.operator_to_letter:
            out.append(info.operator_to_letter[ch])
        else:
            out.append(ch)
    return "".join(out)


def _op_phrase(op_name: str) -> str:
    return {
        "add": "addition",
        "sub": "left minus right when nonnegative",
        "rsub": "right minus left when nonnegative",
        "sub_signed": "signed left minus right",
        "rsub_signed": "signed right minus left",
        "absdiff": "absolute difference",
        "neg_absdiff": "negative absolute difference",
        "mul": "multiplication",
        "mul_p1": "multiply then add one",
        "mul_m1": "multiply then subtract one",
        "add_m1": "add then subtract one",
        "add_p1": "add then add one",
        "concat_fwd": "concatenate left then right",
        "concat_rev": "concatenate right then left",
    }.get(op_name, op_name.replace("_", " "))


MODEL_OP_PHRASES = {
    # Phrasing harvested from Kien baseline eq-symbol traces such as
    # 0133bcec, 02a04b59, 065abaf6 and adapted for the documented ops.
    "add": "addition",
    "sub": "subtraction",
    "rsub": "reversed subtraction",
    "sub_signed": "signed subtraction",
    "rsub_signed": "signed reversed subtraction",
    "absdiff": "absolute difference",
    "neg_absdiff": "negative absolute difference",
    "mul": "multiplication",
    "mul_p1": "multiplication, then add one",
    "mul_m1": "multiplication, then subtract one",
    "mul_p2": "multiplication, then add two",
    "mul_m2": "multiplication, then subtract two",
    "mul_half": "half the product",
    "mul_double": "double the product",
    "mul_plus_a": "multiplication, then add the left number",
    # These four fallback labels are intentionally historical: the published
    # Arm A adapter was trained on these exact strings before the phrase table
    # was expanded. Keep them for byte-level reproduction of the reported CSV.
    "mul_plus_b": "mul plus b",
    "mul_minus_a": "multiplication, then subtract the left number",
    "mul_minus_b": "multiplication, then subtract the right number",
    "add_m1": "addition, then subtract one",
    "add_p1": "addition, then add one",
    "add_m2": "addition, then subtract two",
    "add_p2": "addition, then add two",
    "sub_m1": "subtraction, then subtract one",
    "sub_p1": "subtraction, then add one",
    "rsub_m1": "reversed subtraction, then subtract one",
    "rsub_p1": "reversed subtraction, then add one",
    "absdiff_m1": "absolute difference, then subtract one",
    "absdiff_p1": "absdiff p1",
    "absdiff_m2": "absdiff m2",
    "absdiff_p2": "absolute difference, then add two",
    "concat_fwd": "concatenation",
    "concat_rev": "reverse concatenation",
    "gcd": "greatest common divisor",
    "lcm": "least common multiple",
    "fdiv": "integer division",
    "rdiv": "reversed integer division",
    "mod": "modulus",
    "rmod": "reversed modulus",
    "min": "minimum",
    "max": "maximum",
    "sq_diff": "sq diff",
    "sq_sum": "the square of the sum",
    "a2_plus_b": "the square of the left number, then add the right number",
    "a_plus_b2": "the left number plus the square of the right number",
    "xor": "bitwise xor",
    "band": "bitwise and",
    "bor": "bitwise or",
}
MODEL_PHRASE_TO_OP = {phrase: name for name, phrase in MODEL_OP_PHRASES.items()}


def _model_op_phrase(op_name: str) -> str:
    return MODEL_OP_PHRASES.get(canonical_op_name(op_name), op_name.replace("_", " "))


def _op_family_hint(op_name: str) -> str:
    if op_name.startswith("mul") or op_name in {"sq_diff", "sq_sum", "a2_plus_b", "a_plus_b2"}:
        return "multiplication-like"
    if op_name in {"concat_fwd", "concat_rev"}:
        return "concatenation-like"
    if "mod" in op_name or "div" in op_name or op_name in {"gcd", "lcm"}:
        return "division/modulus-like"
    if "sub" in op_name or "diff" in op_name:
        return "subtraction-like"
    return "addition-like"


def _structure_lines(
    puzzle: Puzzle,
    solution: Solution,
    prefix_info: PrefixInfo | None = None,
) -> list[str]:
    lines: list[str] = []
    by_op: dict[str, list[int]] = {}
    for eq in puzzle.examples:
        if eq.op in solution.op_assignment:
            by_op.setdefault(eq.op, []).append(_body_width_for_result(eq.result, eq.op))
    for op_glyph, widths in sorted(by_op.items(), key=lambda item: item[0]):
        op_name = canonical_op_name(solution.op_assignment[op_glyph])
        label = prefix_info.operator_to_letter[op_glyph] if prefix_info else op_glyph
        width_text = ",".join(str(width) for width in widths)
        lines.append(
            f"Structure for operator {label}: result widths {width_text} from two-digit operands -> {_op_family_hint(op_name)}."
        )
    return lines


def _space_chars(value: str) -> str:
    return " ".join(value)


def _line_pair_text(value: str, symbol_to_letter: dict[str, str], operator_to_letter: dict[str, str]) -> str:
    pairs: list[str] = []
    for ch in value:
        if ch in symbol_to_letter:
            pairs.append(f"{ch} -> {symbol_to_letter[ch]}")
        elif ch in operator_to_letter:
            pairs.append(f"{ch} -> {operator_to_letter[ch]}")
        else:
            raise PrefixVerificationError(f"missing letter for glyph {ch!r}")
    return ", ".join(pairs)


def _example_header(eq: Equation) -> str:
    return f"  「{eq.left}{eq.op}{eq.right}」 = 「{eq.result}」:"


def _input_conversion_line(eq: Equation, info: PrefixInfo) -> str:
    value = eq.left + eq.op + eq.right
    return (
        f"    input:  {_space_chars(value)} : "
        f"{_line_pair_text(value, info.symbol_to_letter, info.operator_to_letter)} -> "
        f"{_letter_expr_display(eq, info.symbol_to_letter, info.operator_to_letter)}"
    )


def _output_conversion_line(eq: Equation, info: PrefixInfo) -> str:
    return (
        f"    output: {_space_chars(eq.result)} : "
        f"{_line_pair_text(eq.result, info.symbol_to_letter, info.operator_to_letter)} -> "
        f"{_letter_result(eq.result, info.symbol_to_letter, info.operator_to_letter)}"
    )


def _query_conversion_line(puzzle: Puzzle, info: PrefixInfo) -> str:
    eq = puzzle.query
    value = eq.left + eq.op + eq.right
    return (
        f"Converting question \"{value}\": {_space_chars(value)} : "
        f"{_line_pair_text(value, info.symbol_to_letter, info.operator_to_letter)} -> "
        f"{_letter_expr_display(eq, info.symbol_to_letter, info.operator_to_letter)}"
    )


def _remaining_conversion_lines(puzzle: Puzzle, info: PrefixInfo) -> list[str]:
    ok_converted = {(line.example_index, line.kind) for line in info.converted_lines if line.ok}
    needs_any = any(
        (index, "input") not in ok_converted or (index, "output") not in ok_converted
        for index in range(len(puzzle.examples))
    )
    if not needs_any:
        return []

    lines: list[str] = []
    if "Converting examples to letter form:" not in info.kept_prefix:
        lines.extend([CONVERSION_MARKER, ""])
    for index, eq in enumerate(puzzle.examples):
        input_done = (index, "input") in ok_converted
        output_done = (index, "output") in ok_converted
        if input_done and output_done:
            continue
        header = _example_header(eq)
        if header not in info.kept_prefix:
            lines.append(header)
        if not input_done:
            lines.append(_input_conversion_line(eq, info))
        if not output_done:
            lines.append(_output_conversion_line(eq, info))
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def recut_prefix_info_to_letter_table(info: PrefixInfo) -> PrefixInfo:
    marker = "Converting examples to letter form:"
    offset = info.kept_prefix.find(marker)
    if offset >= 0:
        kept = info.kept_prefix[:offset].rstrip() + "\n\n"
    else:
        seen_symbols: set[str] = set()
        seen_ops: set[str] = set()
        in_operator_section = False
        best_end = 0
        for _start, end, raw_line in _line_spans(info.kept_prefix):
            stripped = raw_line.strip()
            lower = stripped.lower()
            if stripped == "Operators:":
                in_operator_section = True
                best_end = end
                continue
            if stripped == "Operands:":
                in_operator_section = False
                continue
            match = TABLE_RE.match(raw_line.rstrip("\r\n"))
            if not match:
                continue
            glyph = match.group("glyph").strip()
            label = match.group("label")
            if in_operator_section or glyph in info.operator_to_letter:
                if info.operator_to_letter.get(glyph) == label and glyph not in seen_ops:
                    seen_ops.add(glyph)
                    best_end = end
            elif info.symbol_to_letter.get(glyph) == label and glyph not in seen_symbols:
                seen_symbols.add(glyph)
                best_end = end
        if not best_end:
            return info
        kept = info.kept_prefix[:best_end].rstrip() + "\n\n"
    if kept == info.kept_prefix:
        return info
    return PrefixInfo(
        valid=info.valid,
        cut_offset=info.cut_offset - (len(info.kept_prefix) - len(kept)),
        kept_prefix=kept,
        symbol_to_letter=dict(info.symbol_to_letter),
        operator_to_letter=dict(info.operator_to_letter),
        converted_lines=[],
        cut_reason=f"{info.cut_reason}; recut_to_letter_table",
        next_expected_line="",
            original_prediction=info.original_prediction,
            byte_continuity_end=info.cut_offset,
            byte_continuity_checks=(),
        )


CONVERSION_MARKER = "Converting examples to letter form:"


def _conversion_header_block(text: str, start: int) -> tuple[int, int, str]:
    end = text.find("\n", start)
    if end < 0:
        end = len(text)
    else:
        end += 1
    pos = end
    while pos < len(text):
        next_end = text.find("\n", pos)
        if next_end < 0:
            next_end = len(text)
        else:
            next_end += 1
        raw = text[pos:next_end]
        if raw.strip():
            break
        end = next_end
        pos = next_end
    return start, end, text[start:end]


def conversion_header_byte_target(info: PrefixInfo) -> tuple[int, int, str] | None:
    if not info.original_prediction:
        return None
    start = info.original_prediction.find(CONVERSION_MARKER, info.cut_offset)
    if start < 0:
        return None
    if start - info.cut_offset > 1000:
        return None
    return _conversion_header_block(info.original_prediction, start)


def add_conversion_header_byte_continuity(puzzle: Puzzle, info: PrefixInfo) -> PrefixInfo:
    completed, _ = complete_prefix_info_with_definitions(puzzle, info)
    conversion_lines = _remaining_conversion_lines(puzzle, completed)
    if not conversion_lines or conversion_lines[0] != CONVERSION_MARKER:
        return info
    target = conversion_header_byte_target(info)
    if target is None:
        return info
    start, end, _block = target
    check = ByteContinuityCheck("conversion_header", start, end, 0)
    if check in info.byte_continuity_checks:
        return info
    return PrefixInfo(
        valid=info.valid,
        cut_offset=info.cut_offset,
        kept_prefix=info.kept_prefix,
        symbol_to_letter=dict(info.symbol_to_letter),
        operator_to_letter=dict(info.operator_to_letter),
        converted_lines=info.converted_lines,
        cut_reason=info.cut_reason,
        next_expected_line=info.next_expected_line,
        original_prediction=info.original_prediction,
        byte_continuity_end=info.byte_continuity_end,
        byte_continuity_checks=(*info.byte_continuity_checks, check),
    )


def byte_continuity_target(info: PrefixInfo) -> str:
    if not info.original_prediction:
        return ""
    end = info.byte_continuity_end if info.byte_continuity_end is not None else info.cut_offset
    if end <= info.cut_offset:
        return ""
    return info.original_prediction[info.cut_offset : end]


def verify_prefix_byte_continuity(info: PrefixInfo, completion: str) -> dict[str, Any]:
    target = byte_continuity_target(info)
    checked = False
    chars = 0
    lines = 0
    names: list[str] = []
    if target:
        expected = info.original_prediction[: info.cut_offset + len(target)]
        actual = (info.kept_prefix + completion)[: len(expected)]
        if actual != expected:
            first = next((idx for idx, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
            raise TraceVerificationError(
                "byte continuity mismatch at offset "
                f"{first}: expected {expected[first:first + 40]!r}, got {actual[first:first + 40]!r}"
            )
        checked = True
        chars += len(target)
        lines += len(target.splitlines())
        names.append("cut_continuity")
    for check in info.byte_continuity_checks:
        expected = info.original_prediction[check.original_start : check.original_end]
        actual = completion[check.completion_start : check.completion_start + len(expected)]
        if actual != expected:
            first = next((idx for idx, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
            raise TraceVerificationError(
                f"byte continuity mismatch for {check.name} at relative offset "
                f"{first}: expected {expected[first:first + 40]!r}, got {actual[first:first + 40]!r}"
            )
        checked = True
        chars += len(expected)
        lines += len(expected.splitlines())
        names.append(check.name)
    return {"checked": checked, "chars": chars, "lines": lines, "checks": names}


def _conversion_style_definition_lines(definition_lines: list[str]) -> list[str]:
    if not definition_lines:
        return []
    lines = ["Converting remaining symbols:"]
    for item in definition_lines:
        glyph, label = [part.strip() for part in item.split("->", 1)]
        lines.append(f"    symbol: {_space_chars(glyph)} : {glyph} -> {label} -> {label}")
    return lines


def _check_payload(
    eq: Equation,
    solution: Solution,
    expected: str,
    label_result: str | None = None,
) -> dict[str, Any]:
    op_name = canonical_op_name(solution.op_assignment[eq.op])
    width = _body_width_for_result(expected, eq.op)
    if op_name in SPECIAL_OPS:
        encoded = _concat_result(eq, op_name, overlap=solution.mode.name == "pure_concat_overlap")
        if encoded is None:
            raise TraceVerificationError(f"could not compute special op {op_name} for {eq}")
        return {
            "op": op_name,
            "left_digits": eq.left,
            "right_digits": eq.right,
            "left": 0,
            "right": 0,
            "raw": 0,
            "encoded": label_result if label_result is not None else encoded,
            "glyph_encoded": encoded,
            "expected": label_result if label_result is not None else expected,
            "glyph_expected": expected,
        }
    left_digits, left_value = _pair_value(eq.left, solution.mapping, solution.mode)
    right_digits, right_value = _pair_value(eq.right, solution.mapping, solution.mode)
    pseudo = _pseudo_puzzle(eq)
    encoded, raw = _apply_operation_to_query(pseudo, Solution(solution.mapping, {eq.op: op_name}, solution.mode), width)
    if encoded is None or raw is None:
        raise TraceVerificationError(f"could not compute {eq}")
    return {
        "op": op_name,
        "left_digits": left_digits,
        "right_digits": right_digits,
        "left": left_value,
        "right": right_value,
        "raw": raw,
        "encoded": label_result if label_result is not None else encoded,
        "glyph_encoded": encoded,
        "expected": label_result if label_result is not None else expected,
        "glyph_expected": expected,
    }


def _format_arith_line(
    prefix: str,
    payload: dict[str, Any],
    prefix_info: PrefixInfo | None = None,
    candidates: int | None = None,
) -> str:
    encoded = _encoded_letters(payload["glyph_encoded"], prefix_info) if prefix_info else payload["glyph_encoded"]
    expected = _encoded_letters(payload["glyph_expected"], prefix_info) if prefix_info else payload["glyph_expected"]
    candidate_text = f"candidates={candidates}; " if candidates is not None else ""
    return (
        f"{prefix}: {candidate_text}op={payload['op']}; left={payload['left']}; right={payload['right']}; "
        f"raw={payload['raw']}; encoded={encoded}; expected={expected}."
    )


def _calc_phrase(payload: dict[str, Any]) -> str:
    op_name = canonical_op_name(payload["op"])
    left = int(payload["left"])
    right = int(payload["right"])
    raw = int(payload["raw"])
    if op_name == "add":
        return f"{left} + {right} = {raw}"
    if op_name == "add_p1":
        return f"{left} + {right} = {left + right}, then +1 gives {raw}"
    if op_name == "add_m1":
        return f"{left} + {right} = {left + right}, then -1 gives {raw}"
    if op_name == "add_p2":
        return f"{left} + {right} = {left + right}, then +2 gives {raw}"
    if op_name == "add_m2":
        return f"{left} + {right} = {left + right}, then -2 gives {raw}"
    if op_name == "mul":
        return f"{left} x {right} = {raw}"
    if op_name == "mul_p1":
        return f"{left} x {right} = {left * right}, so with +1 we get {raw}"
    if op_name == "mul_m1":
        return f"{left} x {right} = {left * right}, so with -1 we get {raw}"
    if op_name == "mul_p2":
        return f"{left} x {right} = {left * right}, so with +2 we get {raw}"
    if op_name == "mul_m2":
        return f"{left} x {right} = {left * right}, so with -2 we get {raw}"
    if op_name == "sub":
        return f"{left} - {right} = {raw}"
    if op_name == "rsub":
        return f"{right} - {left} = {raw}"
    if op_name == "sub_signed":
        return f"{left} - {right} = {raw}"
    if op_name == "rsub_signed":
        return f"{right} - {left} = {raw}"
    if op_name == "absdiff":
        return f"|{left} - {right}| = {raw}"
    if op_name == "neg_absdiff":
        return f"-|{left} - {right}| = {raw}"
    if op_name == "mod":
        return f"{left} mod {right} = {raw}"
    if op_name == "rmod":
        return f"{right} mod {left} = {raw}"
    if op_name == "fdiv":
        return f"{left} // {right} = {raw}"
    if op_name == "rdiv":
        return f"{right} // {left} = {raw}"
    if op_name == "gcd":
        return f"gcd({left}, {right}) = {raw}"
    if op_name == "lcm":
        return f"lcm({left}, {right}) = {raw}"
    if op_name == "min":
        return f"min({left}, {right}) = {raw}"
    if op_name == "max":
        return f"max({left}, {right}) = {raw}"
    if op_name == "xor":
        return f"{left} xor {right} = {raw}"
    if op_name == "band":
        return f"{left} and {right} = {raw}"
    if op_name == "bor":
        return f"{left} or {right} = {raw}"
    if op_name in NUMERIC_OPERATIONS:
        return f"{_model_op_phrase(op_name)} gives {raw}"
    if op_name == "concat_fwd":
        return f"concatenation({payload['left_digits']}, {payload['right_digits']}) = {payload['left_digits']} || {payload['right_digits']} = {payload['glyph_encoded']}"
    if op_name == "concat_rev":
        return f"reverse concatenation({payload['left_digits']}, {payload['right_digits']}) = {payload['right_digits']} || {payload['left_digits']} = {payload['glyph_encoded']}"
    return f"{_model_op_phrase(op_name)} gives {payload['glyph_encoded']}"


def _natural_arith_line(
    prefix: str,
    eq: Equation,
    solution: Solution,
    expected: str,
    prefix_info: PrefixInfo,
    relation: str = "matching",
) -> str:
    payload = _check_payload(eq, solution, expected)
    left_letters = "".join(prefix_info.symbol_to_letter[ch] for ch in eq.left)
    right_letters = "".join(prefix_info.symbol_to_letter[ch] for ch in eq.right)
    op_label = prefix_info.operator_to_letter[eq.op]
    op_name = canonical_op_name(payload["op"])
    encoded = _encoded_letters(payload["glyph_encoded"], prefix_info)
    expected_letters = _encoded_letters(payload["glyph_expected"], prefix_info)
    if op_name == "concat_fwd":
        return (
            f"{prefix}: concatenation({left_letters}, {right_letters}) = "
            f"{left_letters} || {right_letters} = {encoded}, {relation} {expected_letters}."
        )
    if op_name == "concat_rev":
        return (
            f"{prefix}: reverse concatenation({left_letters}, {right_letters}) = "
            f"{right_letters} || {left_letters} = {encoded}, {relation} {expected_letters}."
        )
    trying = f"trying {_model_op_phrase(op_name)}: " if relation == "not matching" else ""
    return (
        f"{prefix}: {trying}{left_letters} {op_label} {right_letters} means {payload['left']} and {payload['right']}; "
        f"{_calc_phrase(payload)}; encoded as {encoded}, {relation} {expected_letters}."
    )


def _ranked_anchor_equations(puzzle: Puzzle, solution: Solution) -> list[tuple[int, Equation, str, int]]:
    op_chars = puzzle.example_ops | {puzzle.query.op}
    ranked: list[tuple[int, Equation, str, int]] = []
    for index, eq in enumerate(puzzle.examples, 1):
        if not _equation_usable(eq, op_chars):
            continue
        op_name = canonical_op_name(solution.op_assignment.get(eq.op, ""))
        if not op_name:
            continue
        try:
            count = len(_equation_candidates(eq, op_name, solution.mode))
        except Exception:
            count = 999999
        ranked.append((index, eq, op_name, count))
    ranked.sort(key=lambda item: (item[3], item[0]))
    return ranked


def _candidate_reject_payloads(eq: Equation, solution: Solution, correct_op: str, limit: int = 4) -> list[dict[str, Any]]:
    attempts: list[str] = []
    families = [
        "mul",
        "mul_p1",
        "mul_m1",
        "add",
        "add_p1",
        "add_m1",
        "sub",
        "rsub",
        "sub_signed",
        "rsub_signed",
        "absdiff",
        "concat_fwd",
        "concat_rev",
        "mod",
        "rmod",
    ]
    for name in families:
        canonical = canonical_op_name(name)
        if canonical != correct_op and canonical not in attempts:
            attempts.append(canonical)

    rejects: list[dict[str, Any]] = []
    for op_name in attempts:
        try:
            payload = _check_payload(eq, Solution(solution.mapping, {eq.op: op_name}, solution.mode), eq.result)
        except Exception:
            continue
        if payload["glyph_encoded"] != payload["glyph_expected"]:
            rejects.append(payload)
        if len(rejects) >= limit:
            break
    return rejects


def _narrowing_lines(
    puzzle: Puzzle,
    solution: Solution,
    prefix_info: PrefixInfo | None = None,
) -> tuple[list[str], dict[str, Any]]:
    ranked = _ranked_anchor_equations(puzzle, solution)
    if not ranked:
        return ["Narrowing: no usable example equations remain after operator-contaminated results are skipped."], {
            "anchor_example": None,
            "anchor_candidate_count": 0,
            "narrowing_lines": 0,
        }

    lines: list[str] = []
    index, eq, op_name, count = ranked[0]
    lines.append(f"Anchor: E{index} is the most constraining usable equation with {count} surviving candidates.")
    if count <= 5:
        for payload in _candidate_reject_payloads(eq, solution, op_name):
            lines.append(_format_arith_line(f"Narrow E{index} reject", payload, prefix_info))
        fit_payload = _check_payload(eq, solution, eq.result)
        lines.append(_format_arith_line(f"Narrow E{index} fit", fit_payload, prefix_info, candidates=count))
    else:
        first_payload = _check_payload(eq, solution, eq.result)
        lines.append(_format_arith_line(f"Narrow E{index} crosscheck", first_payload, prefix_info, candidates=count))
        if len(ranked) > 1:
            second_index, second_eq, _second_op, second_count = ranked[1]
            second_payload = _check_payload(second_eq, solution, second_eq.result)
            lines.append(
                _format_arith_line(
                    f"Narrow E{second_index} crosscheck",
                    second_payload,
                    prefix_info,
                    candidates=second_count,
                )
            )
    return lines, {
        "anchor_example": index,
        "anchor_candidate_count": count,
        "narrowing_lines": sum(1 for line in lines if line.startswith("Narrow E")),
    }


def _verification_lines(
    puzzle: Puzzle,
    solution: Solution,
    prefix_info: PrefixInfo | None = None,
) -> list[str]:
    lines: list[str] = []
    op_chars = puzzle.example_ops | {puzzle.query.op}
    for index, eq in enumerate(puzzle.examples, 1):
        if not _equation_usable(eq, op_chars):
            lines.append(
                f"Skip E{index}: its result body contains an operator glyph, so I do not use it as a digit equation."
            )
            continue
        if eq.op not in solution.op_assignment:
            lines.append(
                f"Skip E{index}: its operator is not part of the selected query program."
            )
            continue
        payload = _check_payload(eq, solution, eq.result)
        lines.append(_format_arith_line(f"Check E{index}", payload, prefix_info))
    return lines


def _natural_narrowing_lines(
    puzzle: Puzzle,
    solution: Solution,
    prefix_info: PrefixInfo,
) -> tuple[list[str], dict[str, Any]]:
    ranked = _ranked_anchor_equations(puzzle, solution)
    if not ranked:
        return ["The examples with operator-contaminated results are skipped for the digit search."], {
            "anchor_example": None,
            "anchor_candidate_count": 0,
            "narrowing_lines": 0,
        }

    lines: list[str] = []
    index, eq, op_name, count = ranked[0]
    lines.append(f"Let me test candidates against E{index}.")
    if count <= 5:
        for payload in _candidate_reject_payloads(eq, solution, op_name):
            reject_solution = Solution(solution.mapping, {eq.op: payload["op"]}, solution.mode)
            lines.append(_natural_arith_line(f"Narrow E{index} reject", eq, reject_solution, eq.result, prefix_info, "not matching"))
        lines.append(_natural_arith_line(f"Narrow E{index} fit", eq, solution, eq.result, prefix_info, "matching"))
    else:
        lines.append(_natural_arith_line(f"Narrow E{index} crosscheck", eq, solution, eq.result, prefix_info, "matching"))
        if len(ranked) > 1:
            second_index, second_eq, _second_op, _second_count = ranked[1]
            lines.append(
                _natural_arith_line(
                    f"Narrow E{second_index} crosscheck",
                    second_eq,
                    solution,
                    second_eq.result,
                    prefix_info,
                    "matching",
                )
            )
    return lines, {
        "anchor_example": index,
        "anchor_candidate_count": count,
        "narrowing_lines": sum(1 for line in lines if line.startswith("Narrow E")),
    }


def _natural_verification_lines(puzzle: Puzzle, solution: Solution, prefix_info: PrefixInfo) -> list[str]:
    lines: list[str] = []
    op_chars = puzzle.example_ops | {puzzle.query.op}
    for index, eq in enumerate(puzzle.examples, 1):
        if not _equation_usable(eq, op_chars):
            lines.append(f"Skip E{index}: its result contains an operator symbol, so I leave it out of the digit check.")
            continue
        if eq.op not in solution.op_assignment:
            lines.append(f"Skip E{index}: its operator is not needed for the question.")
            continue
        lines.append(_natural_arith_line(f"Check E{index}", eq, solution, eq.result, prefix_info, "matching"))
    return lines


def _converting_back_line(answer_letters: str, answer: str, info: PrefixInfo) -> str:
    letter_to_symbol = info.letter_to_symbol
    parts = []
    for letter in answer_letters:
        glyph = letter_to_symbol.get(letter)
        if glyph is not None:
            parts.append(f"{letter} -> {glyph}")
    return f"  Converting back: {_space_chars(answer_letters)} : {', '.join(parts)} -> {answer}"


def _query_line(puzzle: Puzzle, solution: Solution, prefix_info: PrefixInfo | None = None) -> str:
    if not puzzle.gold:
        raise TraceVerificationError("missing gold answer")
    payload = _check_payload(puzzle.query, solution, puzzle.gold)
    return _format_arith_line("Query", payload, prefix_info)


def _anchor_index(puzzle: Puzzle, solution: Solution) -> int | None:
    ranked = _ranked_anchor_equations(puzzle, solution)
    return ranked[0][0] if ranked else None


def enumerate_witness_programs(
    puzzle: Puzzle,
    limit: int = 16,
    solution_limit: int = 256,
) -> list[WitnessProgram]:
    witnesses: list[WitnessProgram] = []
    seen: set[tuple[str, str, str]] = set()
    if puzzle.query.op not in puzzle.example_ops:
        return witnesses

    def add_solution(solution: Solution) -> None:
        for answer in _possible_query_answers(puzzle, solution, max(2, limit)):
            key = (
                answer,
                canonical_op_name(solution.op_assignment.get(puzzle.query.op, "")),
                solution.mode.describe(),
            )
            if key not in seen:
                seen.add(key)
                witnesses.append(WitnessProgram(*key))

    for solution in _pure_concat_solutions(puzzle, gold_conditioned=False, max_solutions=limit):
        add_solution(solution)
        if len(witnesses) >= limit:
            return witnesses

    op_chars = puzzle.example_ops | {puzzle.query.op}
    equations = [eq for eq in puzzle.examples if _equation_usable(eq, op_chars)]
    for solution in _solve_with_join(puzzle, equations, MODES, max_solutions=solution_limit):
        add_solution(solution)
        if len(witnesses) >= limit:
            break
    return witnesses


def _assignment_letters(solution: Solution, info: PrefixInfo) -> str:
    if not solution.mapping:
        return "no digit assignment is needed for this pure string operation"
    items = []
    for glyph, digit in sorted(solution.mapping.items(), key=lambda kv: info.symbol_to_letter.get(kv[0], kv[0])):
        items.append(f"{info.symbol_to_letter[glyph]}={digit}")
    return ", ".join(items)


def _assignment_glyphs(solution: Solution) -> str:
    if not solution.mapping:
        return "no digit assignment is needed for this pure string operation"
    return ", ".join(f"{glyph}={digit}" for glyph, digit in sorted(solution.mapping.items()))


def render_arm_a(
    puzzle: Puzzle,
    solver_record: dict[str, Any],
    prefix_info: PrefixInfo,
    inclusion_category: str = "",
) -> TraceBuild:
    if not prefix_info.valid:
        raise PrefixVerificationError(prefix_info.cut_reason)
    original_prefix_info = prefix_info
    prefix_info, definition_lines = complete_prefix_info_with_definitions(puzzle, prefix_info)
    solution = solution_from_record(solver_record)
    missing = [glyph for glyph in puzzle.digit_glyphs if glyph not in prefix_info.symbol_to_letter]
    missing_ops = [glyph for glyph in (puzzle.example_ops | {puzzle.query.op}) if glyph not in prefix_info.operator_to_letter]
    if missing or missing_ops:
        raise PrefixVerificationError(f"missing letter notation symbols={missing} ops={missing_ops}")

    query_op = canonical_op_name(solution.op_assignment[puzzle.query.op])
    query_letters = _letter_expr_display(puzzle.query, prefix_info.symbol_to_letter, prefix_info.operator_to_letter)
    answer_letters = _encoded_letters(puzzle.gold or "", prefix_info)
    narrowing, narrowing_meta = _natural_narrowing_lines(puzzle, solution, prefix_info)
    leading_exact = byte_continuity_target(prefix_info)

    lines: list[str] = []
    conversion_header_exact = ""
    conversion_lines = [] if leading_exact else _remaining_conversion_lines(puzzle, prefix_info)
    if conversion_lines and conversion_lines[0] == CONVERSION_MARKER:
        target = conversion_header_byte_target(prefix_info)
        if target is not None:
            _start, _end, conversion_header_exact = target
            conversion_lines = conversion_lines[1:]
            while conversion_lines and conversion_lines[0] == "":
                conversion_lines = conversion_lines[1:]
    lines.extend(conversion_lines)
    extra_definition_lines = _conversion_style_definition_lines(definition_lines)
    if conversion_lines and extra_definition_lines:
        lines.append("")
    lines.extend(extra_definition_lines)
    if conversion_lines:
        lines.append("")
    elif extra_definition_lines:
        lines.append("")
    # Continue the template Kien uses in correct eq-symbol traces such as
    # 0133bcec and 065abaf6; the surgical change is only the true op phrase.
    lines.append("Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.")
    for op_glyph, op_name in sorted(solution.op_assignment.items(), key=lambda kv: prefix_info.operator_to_letter.get(kv[0], kv[0])):
        lines.append(f"Operator {prefix_info.operator_to_letter[op_glyph]}: {_model_op_phrase(canonical_op_name(op_name))}")
    lines.append("")
    if inclusion_category.startswith("guess"):
        lines.append(
            f"The question operator is {prefix_info.operator_to_letter[puzzle.query.op]}, which is unknown. "
            f"The most common operation in these puzzles is {_model_op_phrase(query_op)}, so we try it."
        )
    else:
        lines.append(f"The question operator is {prefix_info.operator_to_letter[puzzle.query.op]}, which is {_model_op_phrase(query_op)}.")
    lines.append(f"Label the examples E1-E{len(puzzle.examples)} in order.")
    lines.extend(_structure_lines(puzzle, solution, prefix_info))
    if solution.mode.semantic_key == "reverse_digits":
        lines.append("For this puzzle, I read each two-letter number right-to-left before applying the operation.")
    if inclusion_category.startswith("unknown"):
        lines.append(f"The examples leave uniqueness unknown, so I use the deduce prior and try {_model_op_phrase(query_op)}.")
    elif inclusion_category.startswith("ambiguous"):
        lines.append(f"The examples are ambiguous, so I use frequency order and keep the higher-prior {_model_op_phrase(query_op)} program.")
    lines.extend(narrowing)
    lines.append(f"The consistent digit assignment is {_assignment_letters(solution, prefix_info)}.")
    lines.extend(_natural_verification_lines(puzzle, solution, prefix_info))
    lines.append(_query_conversion_line(puzzle, prefix_info))
    lines.append(f"Applying to {query_letters}:")
    lines.append("  " + _natural_arith_line("Query", puzzle.query, solution, puzzle.gold or "", prefix_info, "matching"))
    lines.append(_converting_back_line(answer_letters, puzzle.gold or "", prefix_info))
    lines.append("I will now return the answer in \\boxed{}")
    lines.append("The answer in \\boxed is")
    lines.append(f"\\boxed{{{puzzle.gold}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{puzzle.gold}}}")
    suffix = "\n".join(lines)
    prefix_text = leading_exact
    if conversion_header_exact:
        prefix_text += ("" if not prefix_text or prefix_text.endswith(("\n", "\r")) else "\n")
        prefix_text += conversion_header_exact
    trace = prefix_text + ("" if not prefix_text or prefix_text.endswith(("\n", "\r")) else "\n") + suffix
    return TraceBuild(
        trace=trace,
        metadata={
            "mode": solution.mode.describe(),
            "query_op": query_op,
            "anchor_example": narrowing_meta["anchor_example"],
            "anchor_candidate_count": narrowing_meta["anchor_candidate_count"],
            "narrowing_lines": narrowing_meta["narrowing_lines"],
            "answer_letters": answer_letters,
            "prefix_cut_reason": prefix_info.cut_reason,
            "definition_lines": definition_lines,
            "symbol_to_letter": prefix_info.symbol_to_letter,
            "operator_to_letter": prefix_info.operator_to_letter,
            "original_symbol_to_letter": original_prefix_info.symbol_to_letter,
            "original_operator_to_letter": original_prefix_info.operator_to_letter,
        },
    )


def render_arm_b(
    puzzle: Puzzle,
    solver_record: dict[str, Any],
    inclusion_category: str = "",
) -> TraceBuild:
    solution = solution_from_record(solver_record)
    query_op = canonical_op_name(solution.op_assignment[puzzle.query.op])
    narrowing, narrowing_meta = _narrowing_lines(puzzle, solution, None)
    lines: list[str] = [
        "I will solve the symbol equation directly as a digit cipher.",
        f"Label the examples E1-E{len(puzzle.examples)} in order.",
        "Each expression is two encoded digits, one operator, then two encoded digits.",
    ]
    lines.extend(_structure_lines(puzzle, solution, None))
    if solution.mode.semantic_key == "reverse_digits":
        lines.append("In this row, each two-digit number reads right-to-left.")
    if inclusion_category.startswith("guess"):
        lines.append(f"The query operator is unseen, so the calibrated prior selects {_op_phrase(query_op)}.")
    elif inclusion_category.startswith("ambiguous"):
        lines.append(f"Several programs fit the examples; the frequency prior selects {_op_phrase(query_op)}.")
    elif inclusion_category.startswith("unknown"):
        lines.append(f"Uniqueness is unknown within the budget; the top-four prior allows {_op_phrase(query_op)}.")
    lines.extend(narrowing)
    for op_glyph, op_name in sorted(solution.op_assignment.items()):
        lines.append(f"Operator {op_glyph} is {_op_phrase(canonical_op_name(op_name))}.")
    lines.append(f"Digit assignment: {_assignment_glyphs(solution)}.")
    lines.extend(_verification_lines(puzzle, solution, None))
    lines.append(_query_line(puzzle, solution, None))
    lines.append(f"The encoded answer is {puzzle.gold}.")
    lines.append("</think>")
    lines.append(f"\\boxed{{{puzzle.gold}}}")
    trace = "\n".join(lines)
    return TraceBuild(
        trace=trace,
        metadata={
            "mode": solution.mode.describe(),
            "query_op": query_op,
            "anchor_example": narrowing_meta["anchor_example"],
            "anchor_candidate_count": narrowing_meta["anchor_candidate_count"],
            "narrowing_lines": narrowing_meta["narrowing_lines"],
        },
    )


def approximate_token_count(text: str) -> int:
    return math.ceil(len(text) / 3.5)


def _notation_letters(value: str) -> set[str]:
    return set(re.findall(r"[A-Za-z]", value))


def undefined_letter_references(trace: str, prefix_info: PrefixInfo) -> list[str]:
    """Return structured Arm A notation references that lack a prior table definition."""
    defined = set(prefix_info.symbol_to_letter.values()) | set(prefix_info.operator_to_letter.values())
    problems: list[str] = []
    for line_number, line in enumerate(trace.splitlines(), 1):
        stripped = line.strip()
        table_match = TABLE_RE.match(stripped)
        if table_match:
            defined.add(table_match.group("label"))
            continue

        fields: list[str] = []
        if stripped.startswith("Correcting the next conversion:"):
            fields.append(stripped.rsplit(" ", 1)[-1].rstrip("."))
        if stripped.startswith("Structure for operator "):
            fields.append(stripped.split("Structure for operator ", 1)[1].split(":", 1)[0])
        if stripped.startswith("Operator "):
            fields.append(stripped.split("Operator ", 1)[1].split(":", 1)[0].split(" ", 1)[0])
        if stripped.startswith("The question operator is "):
            fields.append(stripped.split("The question operator is ", 1)[1].split(",", 1)[0].strip())
        if (stripped.startswith("Assignment:") or stripped.startswith("The consistent digit assignment is")) and "=" in stripped:
            fields.extend(re.findall(r"\b([A-Za-z])=", stripped))
        if stripped.startswith("Question in letters:"):
            fields.append(stripped.split(":", 1)[1].strip().rstrip("."))
        if stripped.startswith(("input:", "output:", "symbol:")) and "->" in stripped:
            problem = learn_inline_pairs_for_undefined(stripped, defined)
            if problem:
                problems.append(f"line {line_number}: {problem}")
            fields.append(stripped.rsplit("->", 1)[-1].strip().replace(" ", ""))
        if stripped.startswith("Converting question ") and "->" in stripped:
            problem = learn_inline_pairs_for_undefined(stripped, defined)
            if problem:
                problems.append(f"line {line_number}: {problem}")
            fields.append(stripped.rsplit("->", 1)[-1].strip())
        if stripped.startswith("Applying to "):
            fields.append(stripped.split("Applying to ", 1)[1].rstrip(":").strip())
        if stripped.startswith("Encode "):
            fields.append(stripped.split("Encode ", 1)[1].split(" back", 1)[0])
        if stripped.startswith("Converting back:") and "->" in stripped:
            fields.append(stripped.split(":", 1)[1].split(":", 1)[0].strip().replace(" ", ""))
        match = CHECK_RE.match(stripped)
        if match:
            fields.extend([match.group("encoded").strip(), match.group("expected").strip()])
        natural = NATURAL_CHECK_RE.match(stripped) or NATURAL_CONCAT_RE.match(stripped)
        if natural:
            fields.extend(
                [
                    natural.group("left_letters"),
                    natural.group("right_letters"),
                    natural.group("encoded"),
                    natural.group("expected"),
                ]
            )
            if "op_label" in natural.groupdict():
                fields.append(natural.group("op_label"))

        for field in fields:
            missing = sorted(_notation_letters(field) - defined)
            if missing:
                problems.append(f"line {line_number}: {field!r} uses undefined {missing}")
    return problems


def learn_inline_pairs_for_undefined(line: str, defined: set[str]) -> str:
    for match in INLINE_PAIR_RE.finditer(line):
        defined.add(match.group("label"))
    return ""


def _natural_context(trace: str) -> dict[str, Any]:
    assignment: dict[str, int] = {}
    op_labels: dict[str, str] = {}
    for line in trace.splitlines():
        stripped = line.strip()
        if stripped.startswith("The consistent digit assignment is") or stripped.startswith("Assignment:"):
            for label, digit in re.findall(r"\b([A-Za-z])=(\d)\b", stripped):
                assignment[label] = int(digit)
        op_match = re.match(r"^Operator\s+([A-Za-z]):\s+(.+?)\.?$", stripped)
        if op_match:
            phrase = op_match.group(2).rstrip(".")
            op_name = MODEL_PHRASE_TO_OP.get(phrase)
            if op_name is None:
                candidate = canonical_op_name(phrase.replace(" ", "_"))
                if candidate in NUMERIC_OPERATIONS or candidate in SPECIAL_OPS:
                    op_name = candidate
            if op_name:
                op_labels[op_match.group(1)] = op_name
    return {"assignment": assignment, "op_labels": op_labels, "reverse_digits": "right-to-left" in trace}


def _equation_for_natural_line(match: re.Match[str], puzzle: Puzzle) -> Equation:
    if match.group("prefix") == "Query":
        return puzzle.query
    index_text = match.group("index") or match.group("narrow_index")
    if not index_text:
        raise TraceVerificationError("natural line lacks equation index")
    index = int(index_text)
    if index < 1 or index > len(puzzle.examples):
        raise TraceVerificationError(f"natural line bad E index: {index}")
    return puzzle.examples[index - 1]


def _verify_natural_line(line: str, puzzle: Puzzle, prefix_info: PrefixInfo | None, context: dict[str, Any]) -> bool:
    stripped = line.strip()
    concat = NATURAL_CONCAT_RE.match(stripped)
    natural = NATURAL_CHECK_RE.match(stripped)
    match = concat or natural
    if not match:
        return False
    if prefix_info is None:
        raise TraceVerificationError("natural Arm A line needs prefix_info for verification")
    eq = _equation_for_natural_line(match, puzzle)
    expected = puzzle.gold if match.group("prefix") == "Query" else eq.result
    if expected is None:
        raise TraceVerificationError("missing expected answer for natural query line")
    op_label = match.group("op_label") if natural else None
    if concat:
        op_name = "concat_fwd" if match.group("phrase") == "concatenation" else "concat_rev"
    else:
        try_phrase = match.group("try_phrase")
        op_name = MODEL_PHRASE_TO_OP.get(try_phrase) if try_phrase else None
        if op_name is None and try_phrase:
            candidate = canonical_op_name(try_phrase.replace(" ", "_"))
            if candidate in NUMERIC_OPERATIONS or candidate in SPECIAL_OPS:
                op_name = candidate
        op_name = op_name or context["op_labels"].get(op_label or "")
        if not op_name:
            raise TraceVerificationError(f"unknown natural op label {op_label!r}: {line}")
    mapping: dict[str, int] = {}
    for glyph, label in prefix_info.symbol_to_letter.items():
        if label in context["assignment"]:
            mapping[glyph] = context["assignment"][label]
    mode = Mode(name="little_endian", reverse_digits=True) if context.get("reverse_digits") else Mode(name="standard")
    payload = _check_payload(eq, Solution(mapping, {eq.op: op_name}, mode), expected)
    encoded = _encoded_letters(payload["glyph_encoded"], prefix_info)
    expected_letters = _encoded_letters(payload["glyph_expected"], prefix_info)
    relation = match.group("relation")
    if concat:
        left_letters = "".join(prefix_info.symbol_to_letter[ch] for ch in eq.left)
        right_letters = "".join(prefix_info.symbol_to_letter[ch] for ch in eq.right)
        first = left_letters if op_name == "concat_fwd" else right_letters
        second = right_letters if op_name == "concat_fwd" else left_letters
        if match.group("first") != first or match.group("second") != second:
            raise TraceVerificationError(f"bad concat order: {line}")
    else:
        if int(match.group("left")) != payload["left"] or int(match.group("right")) != payload["right"]:
            raise TraceVerificationError(f"bad natural operand values: {line}")
        if match.group("calc").strip() != _calc_phrase(payload):
            raise TraceVerificationError(f"bad natural calculation: {line}")
    if match.group("encoded").strip() != encoded or match.group("expected").strip() != expected_letters:
        raise TraceVerificationError(f"bad natural encoding: {line}")
    if relation == "matching" and encoded != expected_letters:
        raise TraceVerificationError(f"natural line says matching but differs: {line}")
    if relation == "not matching" and encoded == expected_letters:
        raise TraceVerificationError(f"natural line says reject but matches: {line}")
    return True


def _verify_check_line(
    line: str,
    puzzle: Puzzle,
    prefix_info: PrefixInfo | None,
    context: dict[str, Any],
) -> None:
    if _verify_natural_line(line, puzzle, prefix_info, context):
        return
    match = CHECK_RE.match(line.strip())
    if not match:
        return
    op_name = canonical_op_name(match.group("op"))
    left = int(match.group("left"))
    right = int(match.group("right"))
    raw = int(match.group("raw"))
    encoded = match.group("encoded").strip()
    expected = match.group("expected").strip()
    is_reject = match.group("narrow_kind") == "reject"
    if op_name not in SPECIAL_OPS:
        fn = NUMERIC_OPERATIONS.get(op_name)
        if fn is None:
            raise TraceVerificationError(f"unknown op in trace: {op_name}")
        computed = fn(left, right)
        if op_name == "neg_absdiff" and computed is not None:
            computed = -computed
        if computed != raw:
            raise TraceVerificationError(f"bad arithmetic line: {line}")
    if is_reject and encoded == expected:
        raise TraceVerificationError(f"reject line unexpectedly matches: {line}")
    if not is_reject and encoded != expected:
        raise TraceVerificationError(f"encoded mismatch line: {line}")


def verify_rendered_trace(
    trace: str,
    puzzle: Puzzle,
    prefix_info: PrefixInfo | None = None,
    max_tokens: int = 1300,
) -> dict[str, Any]:
    final = f"\\boxed{{{puzzle.gold}}}"
    if final not in trace:
        raise TraceVerificationError("missing boxed answer")
    if not trace.rstrip().endswith(final):
        raise TraceVerificationError(f"final boxed answer is not gold {puzzle.gold!r}")
    if prefix_info is not None:
        for snippet in ARM_A_BANNED_SNIPPETS:
            if snippet in trace:
                raise TraceVerificationError(f"banned Arm A snippet present: {snippet}")
    label_seen = False
    narrowing_lines = 0
    context = _natural_context(trace)
    for line in trace.splitlines():
        stripped = line.strip()
        intro = LABEL_INTRO_RE.match(stripped)
        if intro:
            if int(intro.group("count")) != len(puzzle.examples):
                raise TraceVerificationError("E-label introduction count does not match examples")
            label_seen = True
            continue
        if re.search(r"\bE\d+\b", line) and not label_seen:
            raise TraceVerificationError(f"E-label used before introduction: {line}")
        if stripped.startswith("Narrow E"):
            narrowing_lines += 1
        _verify_check_line(line, puzzle, prefix_info, context)
    if narrowing_lines == 0:
        raise TraceVerificationError("missing narrowing lines")
    tokens = approximate_token_count(trace)
    if tokens > max_tokens:
        raise TraceVerificationError(f"estimated token count {tokens} > {max_tokens}")
    return {"estimated_tokens": tokens, "chars": len(trace), "boxed": puzzle.gold or "", "narrowing_lines": narrowing_lines}


def trace_completion_with_prefix(prefix_info: PrefixInfo, continuation: str) -> str:
    if prefix_info.kept_prefix and not prefix_info.kept_prefix.endswith(("\n", "\r")):
        return "\n" + continuation
    return continuation
