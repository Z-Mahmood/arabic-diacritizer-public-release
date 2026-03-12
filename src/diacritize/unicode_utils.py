"""Arabic Unicode utilities: normalization, diacritic stripping/extraction/application.

Core invariant:
    apply_diacritics(strip_diacritics(text), extract_diacritics(text)) == normalize(text)

All functions expect and produce NFKC-normalized text.
"""

import unicodedata

from diacritize.config import DIACRITIC_SET, SHADDA

# Arabic letter Unicode range: U+0621 – U+064A
_ARABIC_LETTER_MIN = 0x0621
_ARABIC_LETTER_MAX = 0x064A


def normalize(text: str) -> str:
    """NFKC-normalize Arabic text.

    Ensures consistent combining character ordering and decomposes
    compatibility characters.
    """
    return unicodedata.normalize('NFKC', text)


def is_diacritic(char: str) -> bool:
    """Return True if char is an Arabic diacritic (tashkeel mark)."""
    return char in DIACRITIC_SET


def is_arabic_letter(char: str) -> bool:
    """Return True if char is an Arabic base letter (not a diacritic)."""
    return _ARABIC_LETTER_MIN <= ord(char) <= _ARABIC_LETTER_MAX


def strip_diacritics(text: str) -> str:
    """Remove all diacritics from Arabic text, preserving base characters.

    Normalize first, then filter out diacritic characters.
    """
    text = normalize(text)
    return "".join(ch for ch in text if not is_diacritic(ch))


def extract_diacritics(text: str) -> list[str]:
    """Extract the diacritic label for each base character in text.

    Walks the NFKC-normalized text character by character. For each base
    character (non-diacritic), collects all following diacritics into a
    single label string. Shadda compounds (e.g. shadda+fatha) become a
    multi-char string.

    Returns a list with one entry per base character. Non-Arabic characters
    (spaces, punctuation, digits) get "" since they never carry diacritics.

    IMPORTANT: After collecting diacritics for a character, you need to
    ensure shadda always comes first in compound diacritics (see docstring
    of _normalize_diacritic_order for why).
    """

    normalized = unicodedata.normalize('NFKC', text)
    labels = []
    current_diacritics = ""
    for char in normalized:
        if is_diacritic(char):
            current_diacritics += char
        else:
            if labels:
                labels[-1] = _normalize_diacritic_order(current_diacritics)
            labels.append("")
            current_diacritics = ""

    if labels and current_diacritics:
        labels[-1] = _normalize_diacritic_order(current_diacritics)

    return labels

def _normalize_diacritic_order(diacritics: str) -> str:
    """Ensure shadda always comes first in compound diacritics.

    NFKC sorts by Canonical Combining Class (CCC). Vowels like fatha
    have CCC 27-32, while shadda has CCC 33. This means NFKC puts
    vowels BEFORE shadda — but our LABEL_MAP expects shadda first.

    This function reorders so shadda leads in compound diacritics.
    """
    if SHADDA in diacritics and len(diacritics) > 1:
        return SHADDA + diacritics.replace(SHADDA, "")
    return diacritics


def apply_diacritics(stripped: str, labels: list[str]) -> str:
    """Re-apply diacritics to stripped text using a label list.

    Args:
        stripped: Text with all diacritics removed.
        labels: List of diacritic strings, one per character in stripped.
                Each entry is "" (no diacritic) or a diacritic string
                like "\\u064e" or "\\u0651\\u064e".

    Returns:
        NFKC-normalized diacritized text.

    Raises:
        ValueError: If len(labels) != len(stripped).
    """
    if len(labels) != len(stripped):
        raise ValueError(
            f"Length mismatch: stripped has {len(stripped)} chars, "
            f"but got {len(labels)} labels"
        )

    parts = []
    for ch, diac in zip(stripped, labels):
        parts.append(ch)
        if diac:
            parts.append(diac)

    return normalize("".join(parts))
