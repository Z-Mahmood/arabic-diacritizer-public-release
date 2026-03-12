"""Evaluation metrics for Arabic diacritization.

Three levels of measurement:
    - DER (Diacritic Error Rate): % of characters with wrong diacritics
    - WER (Word Error Rate): % of words with at least one wrong diacritic
    - Per-diacritic accuracy: breakdown by diacritic type
"""

from diacritize.unicode_utils import normalize, extract_diacritics, strip_diacritics


def diacritic_error_rate(predicted: str, reference: str) -> float:
    """Compute Diacritic Error Rate (DER).

    Compares two diacritized strings character-by-character.
    Both are normalized first, then diacritics are extracted and compared.

    DER = number of characters with wrong diacritics / total characters

    Args:
        predicted: Model's diacritized output.
        reference: Ground truth diacritized text.

    Returns:
        DER as a float between 0.0 (perfect) and 1.0 (all wrong).
        Returns 0.0 if reference has no characters.
    """
    pred_labels = extract_diacritics(normalize(predicted))
    ref_labels = extract_diacritics(normalize(reference))

    total = len(ref_labels)
    if total == 0:
        return 0.0

    wrong = sum(1 for p, r in zip(pred_labels, ref_labels) if p !=
                r)
    return wrong / total


def word_error_rate(predicted: str, reference: str) -> float:
    """Compute Word Error Rate (WER).

    Splits both strings into words, compares each word's diacritics.
    A word is "wrong" if ANY character has the wrong diacritic.

    WER = number of wrong words / total words

    Args:
        predicted: Model's diacritized output.
        reference: Ground truth diacritized text.

    Returns:
        WER as a float between 0.0 and 1.0.
        Returns 0.0 if reference has no words.
    """
    pred_words = normalize(predicted).split()
    ref_words = normalize(reference).split()

    total = len(ref_words)
    if total == 0:
        return 0.0

    wrong = 0
    for pw, rw in zip(pred_words, ref_words):
        if extract_diacritics(pw) != extract_diacritics(rw):
            wrong += 1

    return wrong / total


def per_diacritic_accuracy(predicted: str, reference: str) -> dict[str, float]:
    """Compute accuracy for each diacritic type separately.

    For each diacritic that appears in the reference, what % did the
    model get right? This tells you which diacritics the model struggles with.

    Args:
        predicted: Model's diacritized output.
        reference: Ground truth diacritized text.

    Returns:
        Dict mapping diacritic string → accuracy (0.0 to 1.0).
        Only includes diacritics that appear in the reference.
        Example: {"َ": 0.95, "ُ": 0.88, "ِ": 0.92, ...}
    """
    pred_labels = extract_diacritics(normalize(predicted))
    ref_labels = extract_diacritics(normalize(reference))

    correct: dict[str, int] = {}
    total: dict[str, int] = {}

    for p, r in zip(pred_labels, ref_labels):
        if r == "":
            continue
        total[r] = total.get(r, 0) + 1
        if p == r:
            correct[r] = correct.get(r, 0) + 1

    return {d: correct.get(d, 0) / total[d] for d in total}
