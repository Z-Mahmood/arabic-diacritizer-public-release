"""High-level Arabic diacritization pipeline: sentence cache + BiLSTM fallback."""

from __future__ import annotations

from pathlib import Path

from diacritize.baseline.model import BiLSTMDiacritizer
from diacritize.cache import WordCache
from diacritize.config import DATA_DIR, MODELS_DIR
from diacritize.unicode_utils import strip_diacritics


class Diacritizer:
    """Arabic diacritization: sentence cache + BiLSTM fallback.

    Usage::

        d = Diacritizer.from_pretrained()
        print(d.diacritize("بسم الله الرحمن الرحيم"))
    """

    def __init__(
        self,
        model: BiLSTMDiacritizer,
        cache: WordCache | None = None,
    ) -> None:
        self.model = model
        self.cache = cache

    def diacritize(self, text: str) -> str:
        """Diacritize Arabic text using cache-first, model-fallback strategy."""
        if not text or not text.strip():
            return text

        stripped = strip_diacritics(text)

        # Try cache first
        if self.cache is not None:
            cache_pred, mask = self.cache.diacritize(stripped)
            if all(mask):
                return cache_pred
        else:
            mask = [False] * len(stripped.split())

        # Model inference on full sentence (context-aware)
        model_pred = self.model.diacritize(stripped)

        # If no cache hits at all, return model prediction directly
        if not any(mask):
            return model_pred

        # Merge: cache where hit, model where miss
        cached_words = self.cache.diacritize(stripped)[0].split()
        model_words = model_pred.split()
        result: list[str] = []
        for i, hit in enumerate(mask):
            if hit and i < len(cached_words):
                result.append(cached_words[i])
            elif i < len(model_words):
                result.append(model_words[i])
            elif i < len(cached_words):
                result.append(cached_words[i])
        return " ".join(result)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | None = None,
        cache_path: str | None = None,
        no_cache: bool = False,
    ) -> Diacritizer:
        """Load model and cache from default or custom paths.

        Args:
            model_path: Path to BiLSTM .pt checkpoint. Defaults to models/bilstm_best.pt.
            cache_path: Path to word_cache.json. Defaults to data/word_cache.json.
            no_cache: If True, skip loading cache (model-only mode).
        """
        model = BiLSTMDiacritizer.from_pretrained(model_path)

        cache = None
        if not no_cache:
            if cache_path:
                cp = Path(cache_path)
            else:
                # Prefer compressed version if available
                cp = DATA_DIR / "word_cache.json.gz"
                if not cp.exists():
                    cp = DATA_DIR / "word_cache.json"
            if cp.exists():
                cache = WordCache(cp)

        return cls(model=model, cache=cache)
