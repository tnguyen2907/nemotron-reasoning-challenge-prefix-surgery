from __future__ import annotations

def task_family(prompt: str | None) -> str:
    # These leading phrases come from the inspected Kaggle train set and cover
    # the six known families: bit, gravity, unit, cipher text, Roman numerals,
    # and equation transformations. The broader fallbacks support tiny tests
    # and any hand-authored rows we create later.
    text = (prompt or "").lower()
    if "secret bit manipulation rule transforms 8-bit binary numbers" in text:
        return "bit_manipulation"
    if "gravitational constant has been secretly changed" in text or "falling distance" in text or "d = 0.5*g*t^2" in text:
        return "gravity"
    if "secret unit conversion is applied to measurements" in text:
        return "unit_conversion"
    if "secret encryption rules are used on text" in text or "decrypt the following text" in text:
        return "cipher_text"
    if "numbers are secretly converted into a different numeral system" in text or "wonderland numeral system" in text:
        return "roman_numeral"
    if "secret set of transformation rules is applied to equations" in text:
        return "equation_transformation"

    # Fallbacks for hand-authored examples and future derived datasets.
    if "bit manipulation" in text or "8-bit binary" in text or "binary numbers" in text:
        return "bit_manipulation"
    if "gravitational" in text or "gravity" in text:
        return "gravity"
    if "unit" in text or "convert" in text:
        return "unit_conversion"
    if "cipher" in text or "encrypt" in text or "decrypt" in text or "wonderland" in text and "letter" in text:
        return "cipher_text"
    if "roman" in text or "numeral system" in text:
        return "roman_numeral"
    if "equation" in text or "solve for" in text or "algebra" in text:
        return "equation_transformation"
    if "cryptarithm" in text or "alphametic" in text:
        return "cryptarithm"
    return "unknown"


def task_variant(prompt: str | None) -> str:
    family = task_family(prompt)
    text = prompt or ""
    if family != "equation_transformation":
        return family
    return "equation_digit_ops" if any(char.isdigit() for char in text) else "equation_symbol_cipher"
