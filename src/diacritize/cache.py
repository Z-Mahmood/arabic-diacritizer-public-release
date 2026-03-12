"""Word and sentence-level diacritization cache with multi-variant support.

Lookup order at inference:
1. Sentence cache — normalized key match, then best stripped-variant match
2. Word cache — same: normalized key, then stripped-variant match
3. Caller handles remaining misses (model fallback)

Multi-variant storage:
    Each normalized key maps to {stripped_variant: diacritized_form}.
    This handles multiple Quran editions (Hafs, Warsh, etc.) and other
    orthographic variants under a single normalized key.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from pathlib import Path

from diacritize.config import DIACRITIC_SET

# Quranic annotation signs (U+06D6-U+06ED) — tajweed marks, pause signs,
# small letter annotations used in Uthmani script but not in training data.
_QURAN_ANNOTATION_SIGNS = frozenset(chr(c) for c in range(0x06D6, 0x06EE))

# Additional combining marks outside the annotation range.
# U+0653 maddah above and U+0670 superscript alef are diacritical.
# U+0656-U+065F are extended Arabic diacritics (subscript alef, inverted
# damma, mark noon ghunna, etc.) that must be stripped for lookup.
# NOTE: U+0654 (hamza above) and U+0655 (hamza below) are deliberately
# KEPT — they participate in NFKC composition (ي+ٔ→ئ, و+ٔ→ؤ, etc.)
# and carry semantic meaning.
_EXTRA_COMBINING = frozenset([
    "\u0653",  # ٓ maddah above
    "\u0670",  # ٰ superscript alef
    *[chr(c) for c in range(0x0656, 0x0660)],  # U+0656-U+065F extended diacritics
])

_ALL_DIACRITICS = DIACRITIC_SET | _QURAN_ANNOTATION_SIGNS | _EXTRA_COMBINING
_STRIP_PATTERN = re.compile(f"[{''.join(_ALL_DIACRITICS)}]")

# Structural markers to remove during normalization
_QURAN_MARKERS = re.compile(
    "[\u06DD"           # end of ayah
    "\u06DE"            # start of rub el hizb ۞
    "\u0660-\u0669"     # Arabic-Indic digits
    "\u00A0\u202F"      # non-breaking spaces around verse markers
    "]"
)


def _normalize_for_lookup(text: str) -> str:
    """Normalize Arabic text for consistent cache lookups.

    More aggressive than the training unicode_utils.normalize() —
    handles alef variants, Quranic markers, tatweel, Farsi yeh.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0622", "\u0627")  # آ → ا
    text = text.replace("\u0623", "\u0627")  # أ → ا
    text = text.replace("\u0625", "\u0627")  # إ → ا
    text = text.replace("\u0671", "\u0627")  # ٱ → ا
    text = text.replace("\u06CC", "\u064A")  # ی → ي
    # tatweel already removed in _strip_for_lookup (before NFKC)
    text = text.replace("\u0640", "")         # safety — catch any remaining
    text = _QURAN_MARKERS.sub("", text)
    return text


def _strip_for_lookup(text: str) -> str:
    """Strip all diacritics, Quranic marks, and structural markers.

    Also decomposes precomposed alef+diacritic characters (آ)
    so the diacritic portion gets stripped consistently.

    Tatweel (U+0640) is removed here (before NFKC in _normalize_for_lookup)
    because it acts as a base character that blocks NFKC composition of
    sequences like ي+ٔ→ئ when tatweel sits between them.

    Structural markers (verse numbers, NBSP, rub el hizb) are stripped
    because they're metadata, not orthography — they pollute variant keys
    when comparing across Quran editions that embed them differently.
    """
    # Decompose precomposed alef variants where the mark is a diacritic:
    #   آ (U+0622) = alef + madda above → ا (madda is stripped below)
    text = text.replace("\u0622", "\u0627")  # آ → ا
    text = text.replace("\u0640", "")         # tatweel — must remove before NFKC
    text = _STRIP_PATTERN.sub("", text)
    text = _QURAN_MARKERS.sub("", text)
    # NFKC ensures consistent composition of remaining combining marks
    # (e.g. ي+ٔ→ئ, و+ٔ→ؤ) that survived stripping. Without this,
    # build-time vs lookup-time variant keys can diverge when the caller
    # pre-applies NFKC via strip_diacritics().
    text = unicodedata.normalize("NFKC", text)
    return text


def clean_for_lookup(text: str) -> str:
    """Remove structural markers (verse numbers, NBSP) before diacritic stripping.

    Call this BEFORE strip_diacritics() when preparing Quranic text for cache
    lookup. This prevents NFKC normalization from converting U+202F to U+0020
    which would create spurious word splits (e.g. صِرَٰطَ → صر ط).
    """
    return _QURAN_MARKERS.sub("", text)


def _make_key(text: str) -> str:
    """Normalized key: strip diacritics then normalize."""
    return _normalize_for_lookup(_strip_for_lookup(text))


def _make_variant_key(text: str) -> str:
    """Stripped variant key: strip diacritics only (preserves orthography)."""
    return _strip_for_lookup(text)


class WordCache:
    """Lookup cache for unambiguous Arabic diacritizations.

    Internal format (multi-variant):
        words:     {normalized_key: {stripped_variant: diacritized, ...}}
        sentences: {normalized_key: {stripped_variant: diacritized, ...}}

    JSON format supports both:
        Old: {"key": "diacritized_string"}
        New: {"key": {"variant1": "diacritized1", "variant2": "diacritized2"}}
    """

    def __init__(self, cache_path: str | Path) -> None:
        """Load cache from JSON file (supports .json and .json.gz)."""
        cache_path = Path(cache_path)
        if cache_path.name.endswith(".gz"):
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
        self._meta: dict = data["meta"]
        self._words = self._load_entries(data.get("words", {}))
        self._sentences = self._load_entries(data.get("sentences", {}), strip_keys=True)

    @staticmethod
    def _load_entries(
        raw: dict, strip_keys: bool = False,
    ) -> dict[str, dict[str, str]]:
        """Load cache entries, handling both old (string) and new (dict) formats."""
        entries: dict[str, dict[str, str]] = {}
        for k, v in raw.items():
            norm_key = _make_key(k).strip() if strip_keys else _make_key(k)
            if norm_key not in entries:
                entries[norm_key] = {}
            if isinstance(v, dict):
                # New format: {stripped_variant: diacritized}
                entries[norm_key].update(v)
            else:
                # Old format: single string value — derive variant key from the key itself
                variant = _make_variant_key(k).strip() if strip_keys else _make_variant_key(k)
                entries[norm_key][variant] = v
        return entries

    def _lookup(
        self, text: str, store: dict[str, dict[str, str]], strip: bool = False,
    ) -> str | None:
        """Look up text in a multi-variant store.

        1. Compute normalized key to find the entry
        2. Try exact match on stripped variant (matches the input's orthography)
        3. Fallback: return any stored variant
        """
        key = _make_key(text).strip() if strip else _make_key(text)
        variants = store.get(key)
        if variants is None:
            return None
        # Try exact stripped-variant match
        variant_key = _make_variant_key(text).strip() if strip else _make_variant_key(text)
        if variant_key in variants:
            return variants[variant_key]
        # Fallback: return first available variant
        return next(iter(variants.values()))

    def lookup_word(self, word: str) -> str | None:
        """Look up a single undiacritized word. Returns diacritized form or None."""
        return self._lookup(word, self._words)

    def lookup_sentence(self, text: str) -> str | None:
        """Look up a full sentence. Returns diacritized form or None."""
        return self._lookup(text, self._sentences, strip=True)

    def diacritize(self, text: str) -> tuple[str, list[bool]]:
        """Diacritize text using sentence-level cache only.

        Returns:
            (diacritized_text, mask) where mask[i] is True if word[i] was resolved
            from cache. If the full sentence matched, all mask values are True.
            Otherwise returns the original text with all-False mask.
        """
        if not text or not text.strip():
            return (text, [])

        # Sentence-level lookup only — word-level cache removed because
        # context-free word lookups hurt accuracy on general text
        sentence_hit = self.lookup_sentence(text)
        if sentence_hit is not None:
            word_count = len(text.split())
            return (sentence_hit, [True] * word_count)

        # No sentence match — let the model handle it
        words = text.split()
        return (text, [False] * len(words))

    @property
    def stats(self) -> dict:
        """Return cache metadata."""
        total_word_variants = sum(len(v) for v in self._words.values())
        total_sent_variants = sum(len(v) for v in self._sentences.values())
        return {
            **self._meta,
            "cached_words": len(self._words),
            "word_variants": total_word_variants,
            "cached_sentences": len(self._sentences),
            "sentence_variants": total_sent_variants,
        }
