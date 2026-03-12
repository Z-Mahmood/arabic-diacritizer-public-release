"""Tests for evaluate.py: DER, WER, and per-diacritic accuracy."""

import pytest

from diacritize.config import FATHA, DAMMA, KASRA, SUKUN, SHADDA
from diacritize.evaluate import diacritic_error_rate, word_error_rate, per_diacritic_accuracy


# ── Test data ──────────────────────────────────────────────────────────────────
# Perfect prediction
REFERENCE = "مُحَمَّدٌ"
PERFECT = "مُحَمَّدٌ"

# One wrong diacritic: first char damma→fatha
ONE_WRONG = "مَحَمَّدٌ"

# All wrong: strip all diacritics (every char gets "" instead of its diacritic)
ALL_WRONG = "محمد"

# Multi-word
REF_MULTI = "بِسْمِ اللَّهِ"
PRED_MULTI_PERFECT = "بِسْمِ اللَّهِ"
PRED_MULTI_ONE_WORD_WRONG = "بِسْمِ اللَهِ"  # dropped shadda on lam


class TestDiacriticErrorRate:
    def test_perfect_prediction(self):
        assert diacritic_error_rate(PERFECT, REFERENCE) == 0.0

    def test_one_wrong_out_of_four(self):
        # 4 base chars, 1 wrong → DER = 0.25
        der = diacritic_error_rate(ONE_WRONG, REFERENCE)
        assert der == pytest.approx(0.25)

    def test_all_wrong(self):
        # All 4 diacritics missing → DER = 1.0
        der = diacritic_error_rate(ALL_WRONG, REFERENCE)
        assert der == pytest.approx(1.0)

    def test_empty_strings(self):
        assert diacritic_error_rate("", "") == 0.0

    def test_no_diacritics_in_either(self):
        assert diacritic_error_rate("بسم", "بسم") == 0.0

    def test_returns_float(self):
        result = diacritic_error_rate(PERFECT, REFERENCE)
        assert isinstance(result, float)


class TestWordErrorRate:
    def test_perfect_prediction(self):
        assert word_error_rate(PRED_MULTI_PERFECT, REF_MULTI) == 0.0

    def test_one_word_wrong(self):
        # 2 words, 1 wrong → WER = 0.5
        wer = word_error_rate(PRED_MULTI_ONE_WORD_WRONG, REF_MULTI)
        assert wer == pytest.approx(0.5)

    def test_single_word_wrong(self):
        wer = word_error_rate(ONE_WRONG, REFERENCE)
        assert wer == pytest.approx(1.0)  # only 1 word, it's wrong

    def test_single_word_perfect(self):
        assert word_error_rate(PERFECT, REFERENCE) == 0.0

    def test_empty_strings(self):
        assert word_error_rate("", "") == 0.0

    def test_all_stripped(self):
        wer = word_error_rate(ALL_WRONG, REFERENCE)
        assert wer == pytest.approx(1.0)


class TestPerDiacriticAccuracy:
    def test_perfect_all_100(self):
        acc = per_diacritic_accuracy(PERFECT, REFERENCE)
        for diac, val in acc.items():
            assert val == pytest.approx(1.0), f"{diac!r} not 1.0"

    def test_one_wrong_damma(self):
        """First char: damma→fatha. Damma accuracy should drop."""
        acc = per_diacritic_accuracy(ONE_WRONG, REFERENCE)
        # Damma appears once in reference (مُ), model predicted fatha instead
        assert acc[DAMMA] == pytest.approx(0.0)
        # Fatha still correct (حَ and مَّ both have fatha component)
        assert acc[FATHA] == pytest.approx(1.0)

    def test_returns_dict(self):
        result = per_diacritic_accuracy(PERFECT, REFERENCE)
        assert isinstance(result, dict)

    def test_only_includes_reference_diacritics(self):
        """Should not include diacritics that don't appear in reference."""
        acc = per_diacritic_accuracy(PERFECT, REFERENCE)
        # REFERENCE has damma, fatha, shadda+fatha, dammatan — not kasra or sukun
        assert KASRA not in acc
        assert SUKUN not in acc

    def test_empty_strings(self):
        acc = per_diacritic_accuracy("", "")
        assert acc == {}
