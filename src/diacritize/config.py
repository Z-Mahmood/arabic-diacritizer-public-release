"""Configuration: Unicode constants, label map, paths, and model defaults."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Arabic Diacritic Unicode Constants ─────────────────────────────────────────
# Combining marks that appear after a base character in Unicode

FATHA = "\u064E"       # فَتْحَة  — short 'a' above
DAMMA = "\u064F"       # ضَمَّة   — short 'u' above
KASRA = "\u0650"       # كَسْرَة  — short 'i' below
SUKUN = "\u0652"       # سُكُون   — no vowel (quiescent)
SHADDA = "\u0651"      # شَدَّة   — consonant gemination
FATHATAN = "\u064B"    # فَتْحَتان — nunation '-an'
DAMMATAN = "\u064C"    # ضَمَّتان — nunation '-un'
KASRATAN = "\u064D"    # كَسْرَتان — nunation '-in'

# All individual diacritics (order matters for consistency)
DIACRITICS = (FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN)

# Set for fast membership testing
DIACRITIC_SET = frozenset(DIACRITICS)

# ── 15-Class Label Map ─────────────────────────────────────────────────────────
# Index 0 = no diacritic, 1-8 = standalone marks, 9-14 = shadda compounds
#
# Shadda compounds: shadda always precedes the vowel in normalized Unicode.
# We treat shadda+vowel as a single class because they attach to the same
# base character and must be predicted together.

LABEL_MAP = {
    "":                 0,   # No diacritic
    FATHA:              1,
    DAMMA:              2,
    KASRA:              3,
    SUKUN:              4,
    SHADDA:             5,   # Shadda alone (rare, but exists)
    FATHATAN:           6,
    DAMMATAN:           7,
    KASRATAN:           8,
    SHADDA + FATHA:     9,   # شَدَّة + فَتْحَة
    SHADDA + DAMMA:     10,  # شَدَّة + ضَمَّة
    SHADDA + KASRA:     11,  # شَدَّة + كَسْرَة
    SHADDA + FATHATAN:  12,  # شَدَّة + فَتْحَتان
    SHADDA + DAMMATAN:  13,  # شَدَّة + ضَمَّتان
    SHADDA + KASRATAN:  14,  # شَدَّة + كَسْرَتان
}

# Reverse map: index → diacritic string
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

NUM_CLASSES = len(LABEL_MAP)  # 15

# ── Model Defaults ───────────────────────────────────────────────────────────

BASELINE_DEFAULTS = {
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 3,
    "dropout": 0.3,
    "batch_size": 128,
    "max_seq_len": 256,
    "learning_rate": 1e-3,
    "epochs": 20,
    "grad_clip": 1.0,
}
