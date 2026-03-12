"""Tests for unicode_utils.py: Arabic diacritic manipulation.

Uses real Arabic text to verify correctness. The core invariant tested:
    apply_diacritics(strip_diacritics(text), extract_diacritics(text)) == normalize(text)
"""

import pytest
import unicodedata

from diacritize.config import (
    FATHA,
    DAMMA,
    KASRA,
    SUKUN,
    SHADDA,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
)
from diacritize.unicode_utils import (
    normalize,
    is_diacritic,
    is_arabic_letter,
    strip_diacritics,
    extract_diacritics,
    apply_diacritics,
)


# ── Test data ──────────────────────────────────────────────────────────────────
# Bismillah: بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
BISMILLAH = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
BISMILLAH_STRIPPED = "بسم الله الرحمن الرحيم"

# Simpler test: محمد with known diacritics
# مُحَمَّدٌ = mim+damma, ha+fatha, mim+shadda+fatha, dal+dammatan
MUHAMMAD = "مُحَمَّدٌ"
MUHAMMAD_STRIPPED = "محمد"


class TestNormalize:
    def test_nfkc_output(self):
        text = "بِسْمِ"
        result = normalize(text)
        assert result == unicodedata.normalize("NFKC", text)

    def test_idempotent(self):
        text = normalize(BISMILLAH)
        assert normalize(text) == text

    def test_empty_string(self):
        assert normalize("") == ""


class TestIsDiacritic:
    def test_all_diacritics_recognized(self):
        for d in [FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN]:
            assert is_diacritic(d), f"{d!r} not recognized as diacritic"

    def test_arabic_letter_not_diacritic(self):
        assert not is_diacritic("ب")
        assert not is_diacritic("م")

    def test_space_not_diacritic(self):
        assert not is_diacritic(" ")

    def test_digit_not_diacritic(self):
        assert not is_diacritic("5")


class TestIsArabicLetter:
    def test_common_letters(self):
        for letter in "بتثجحخدذرزسشصضطظعغفقكلمنهوي":
            assert is_arabic_letter(letter), f"{letter} not recognized"

    def test_hamza(self):
        assert is_arabic_letter("ء")  # U+0621 — start of range

    def test_alef_variants(self):
        assert is_arabic_letter("ا")  # alef
        assert is_arabic_letter("أ")  # alef with hamza above
        assert is_arabic_letter("إ")  # alef with hamza below
        assert is_arabic_letter("آ")  # alef with madda

    def test_diacritics_are_not_letters(self):
        assert not is_arabic_letter(FATHA)
        assert not is_arabic_letter(SHADDA)

    def test_latin_not_arabic(self):
        assert not is_arabic_letter("a")
        assert not is_arabic_letter("Z")


class TestStripDiacritics:
    def test_bismillah(self):
        result = strip_diacritics(BISMILLAH)
        # Remove the superscript alef (U+0670) which is also in bismillah
        # but isn't in our diacritic set — it stays
        assert "بسم" in result
        assert FATHA not in result
        assert KASRA not in result
        assert SHADDA not in result

    def test_muhammad(self):
        assert strip_diacritics(MUHAMMAD) == MUHAMMAD_STRIPPED

    def test_already_stripped(self):
        text = "بسم الله"
        assert strip_diacritics(text) == text

    def test_empty_string(self):
        assert strip_diacritics("") == ""

    def test_preserves_spaces_and_punctuation(self):
        text = f"أ{FATHA} ب{DAMMA}."
        result = strip_diacritics(text)
        assert " " in result
        assert "." in result

    def test_no_diacritics_remain(self):
        result = strip_diacritics(BISMILLAH)
        for ch in result:
            assert not is_diacritic(ch), f"Diacritic {ch!r} still in result"


class TestExtractDiacritics:
    def test_muhammad_labels(self):
        """مُحَمَّدٌ → mim+damma, ha+fatha, mim+shadda+fatha, dal+dammatan"""
        labels = extract_diacritics(MUHAMMAD)
        assert len(labels) == 4  # 4 base characters
        assert labels[0] == DAMMA           # مُ
        assert labels[1] == FATHA           # حَ
        assert labels[2] == SHADDA + FATHA  # مَّ (shadda compound)
        assert labels[3] == DAMMATAN        # دٌ

    def test_stripped_text_all_empty(self):
        labels = extract_diacritics(MUHAMMAD_STRIPPED)
        assert all(l == "" for l in labels)

    def test_empty_string(self):
        assert extract_diacritics("") == []

    def test_space_gets_empty_label(self):
        text = f"أ{FATHA} ب{DAMMA}"
        labels = extract_diacritics(text)
        # "أ", " ", "ب" → 3 base chars
        assert len(labels) == 3
        assert labels[1] == ""  # space has no diacritic

    def test_shadda_alone(self):
        """Shadda without a following vowel."""
        text = f"ب{SHADDA}"
        labels = extract_diacritics(text)
        assert labels[0] == SHADDA

    def test_all_single_diacritics(self):
        singles = [FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN]
        for d in singles:
            text = f"ب{d}"
            labels = extract_diacritics(text)
            assert labels == [d], f"Failed for {d!r}"

    def test_shadda_compounds(self):
        vowels = [FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN]
        for v in vowels:
            text = normalize(f"ب{SHADDA}{v}")
            labels = extract_diacritics(text)
            assert labels == [SHADDA + v], f"Failed for shadda+{v!r}"


class TestApplyDiacritics:
    def test_muhammad_roundtrip(self):
        stripped = strip_diacritics(MUHAMMAD)
        labels = extract_diacritics(MUHAMMAD)
        restored = apply_diacritics(stripped, labels)
        assert restored == normalize(MUHAMMAD)

    def test_empty_labels_returns_stripped(self):
        text = "بسم"
        labels = ["", "", ""]
        assert apply_diacritics(text, labels) == text

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            apply_diacritics("بسم", ["", ""])

    def test_empty_string(self):
        assert apply_diacritics("", []) == ""


class TestRoundtrip:
    """The core invariant: strip + extract + apply == normalize."""

    @pytest.mark.parametrize(
        "text",
        [
            MUHAMMAD,
            BISMILLAH,
            f"الحَمْدُ لِلَّهِ",          # Alhamdulillah
            f"لَا إِلَٰهَ إِلَّا اللَّهُ",  # La ilaha illa Allah
            f"سُبْحَانَ اللَّهِ",          # SubhanAllah
            "بسم",                          # No diacritics
            "",                              # Empty
        ],
        ids=[
            "muhammad",
            "bismillah",
            "alhamdulillah",
            "shahada",
            "subhanallah",
            "no_diacritics",
            "empty",
        ],
    )
    def test_roundtrip(self, text):
        text = normalize(text)
        stripped = strip_diacritics(text)
        labels = extract_diacritics(text)
        restored = apply_diacritics(stripped, labels)
        assert restored == text, (
            f"Roundtrip failed:\n"
            f"  original:  {text!r}\n"
            f"  stripped:  {stripped!r}\n"
            f"  labels:    {labels!r}\n"
            f"  restored:  {restored!r}"
        )
