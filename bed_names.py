import re

_BED_PATTERN = re.compile(r'\b(U[1-8]|JJ-\d{1,2}|CC\d{1,2})\b')


def normalize_bed(short: str) -> str:
    """Convert short-form bed name to long form matching the xlsx convention."""
    if short.startswith("U"):
        return f"Upstairs-{short[1:]}"
    if short.startswith("CC"):
        return f"ChickenCoop-{short[2:]}"
    return short  # JJ-N is already canonical in both formats


def extract_beds(text: str) -> list[str]:
    """Return sorted list of unique normalized bed names mentioned in text."""
    raw = _BED_PATTERN.findall(text)
    return sorted(set(normalize_bed(b) for b in raw))
