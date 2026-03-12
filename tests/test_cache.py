"""Tests for the word + sentence diacritization cache."""

import json
import pytest
from pathlib import Path

from diacritize.cache import WordCache


@pytest.fixture
def cache_file(tmp_path: Path) -> Path:
    """Create a small test cache fixture."""
    data = {
        "meta": {
            "threshold": 0.995,
            "total_word_tokens": 1000,
            "unique_words": 50,
            "word_coverage": 0.85,
            "built_at": "2026-03-03T00:00:00+00:00",
        },
        "words": {
            "بسم": "بِسْمِ",
            "الله": "اللَّهِ",
            "الرحمن": "الرَّحْمَٰنِ",
            "الرحيم": "الرَّحِيمِ",
            "محمد": "مُحَمَّدٌ",
        },
        "sentences": {
            "بسم الله الرحمن الرحيم": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            "الحمد لله رب العالمين": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        },
    }
    path = tmp_path / "test_cache.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def cache(cache_file: Path) -> WordCache:
    return WordCache(cache_file)


class TestCacheLoading:
    def test_loads_from_json(self, cache: WordCache):
        assert cache.stats["threshold"] == 0.995

    def test_counts_entries(self, cache: WordCache):
        assert cache.stats["cached_words"] == 5
        assert cache.stats["cached_sentences"] == 2


class TestWordLookup:
    def test_hit(self, cache: WordCache):
        assert cache.lookup_word("محمد") == "مُحَمَّدٌ"

    def test_miss(self, cache: WordCache):
        assert cache.lookup_word("كتب") is None

    def test_strips_existing_diacritics(self, cache: WordCache):
        assert cache.lookup_word("مُحَمَّدٌ") == "مُحَمَّدٌ"


class TestSentenceLookup:
    def test_hit(self, cache: WordCache):
        result = cache.lookup_sentence("بسم الله الرحمن الرحيم")
        assert result == "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"

    def test_miss(self, cache: WordCache):
        assert cache.lookup_sentence("هذه جملة غير موجودة") is None

    def test_strips_diacritics_before_lookup(self, cache: WordCache):
        result = cache.lookup_sentence("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
        assert result == "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"


class TestDiacritize:
    def test_sentence_cache_takes_priority(self, cache: WordCache):
        text = "بسم الله الرحمن الرحيم"
        result, mask = cache.diacritize(text)
        assert result == "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        assert all(mask)

    def test_sentence_miss_returns_original(self, cache: WordCache):
        """Sentence-level only: partial matches return original text for model fallback."""
        text = "محمد كتب"
        result, mask = cache.diacritize(text)
        assert result == text
        assert mask == [False, False]

    def test_partial_sentence_not_cached(self, cache: WordCache):
        """A substring of a cached sentence is not a sentence match."""
        text = "بسم الله"
        result, mask = cache.diacritize(text)
        assert result == text
        assert mask == [False, False]

    def test_no_cached_words(self, cache: WordCache):
        text = "كتب قرأ"
        result, mask = cache.diacritize(text)
        assert result == "كتب قرأ"
        assert mask == [False, False]

    def test_empty_input(self, cache: WordCache):
        result, mask = cache.diacritize("")
        assert result == ""
        assert mask == []

    def test_whitespace_only(self, cache: WordCache):
        result, mask = cache.diacritize("   ")
        assert result == "   "
        assert mask == []

    def test_single_word_not_sentence_matched(self, cache: WordCache):
        """Single words don't match sentence cache — model handles them."""
        result, mask = cache.diacritize("الله")
        assert result == "الله"
        assert mask == [False]
