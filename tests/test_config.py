"""Tests for config.py: verify Unicode constants, label map, and model defaults."""

from diacritize.config import (
    DIACRITICS,
    DIACRITIC_SET,
    FATHA,
    DAMMA,
    KASRA,
    SUKUN,
    SHADDA,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
    LABEL_MAP,
    ID_TO_LABEL,
    NUM_CLASSES,
    BASELINE_DEFAULTS,
    ASSETS_DIR,
    DATA_DIR,
    MODELS_DIR,
)


class TestUnicodeConstants:
    """Verify each constant is the correct Unicode codepoint."""

    def test_fatha(self):
        assert FATHA == "\u064E"

    def test_damma(self):
        assert DAMMA == "\u064F"

    def test_kasra(self):
        assert KASRA == "\u0650"

    def test_sukun(self):
        assert SUKUN == "\u0652"

    def test_shadda(self):
        assert SHADDA == "\u0651"

    def test_fathatan(self):
        assert FATHATAN == "\u064B"

    def test_dammatan(self):
        assert DAMMATAN == "\u064C"

    def test_kasratan(self):
        assert KASRATAN == "\u064D"

    def test_diacritics_tuple_has_8_elements(self):
        assert len(DIACRITICS) == 8

    def test_diacritic_set_matches_tuple(self):
        assert DIACRITIC_SET == frozenset(DIACRITICS)

    def test_all_diacritics_are_combining_marks(self):
        """Arabic diacritics live in the U+064x range (combining marks)."""
        for d in DIACRITICS:
            assert 0x064B <= ord(d) <= 0x0652, f"{d!r} (U+{ord(d):04X}) out of range"


class TestLabelMap:
    """Verify the 15-class label map structure."""

    def test_num_classes_is_15(self):
        assert NUM_CLASSES == 15

    def test_label_map_has_15_entries(self):
        assert len(LABEL_MAP) == 15

    def test_no_diacritic_is_index_0(self):
        assert LABEL_MAP[""] == 0

    def test_single_diacritics_indices_1_to_8(self):
        singles = [FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN]
        for d in singles:
            assert 1 <= LABEL_MAP[d] <= 8

    def test_shadda_compounds_indices_9_to_14(self):
        compounds = [
            SHADDA + FATHA,
            SHADDA + DAMMA,
            SHADDA + KASRA,
            SHADDA + FATHATAN,
            SHADDA + DAMMATAN,
            SHADDA + KASRATAN,
        ]
        for c in compounds:
            assert 9 <= LABEL_MAP[c] <= 14

    def test_all_indices_unique(self):
        values = list(LABEL_MAP.values())
        assert len(values) == len(set(values))

    def test_indices_are_contiguous_0_to_14(self):
        assert set(LABEL_MAP.values()) == set(range(15))

    def test_id_to_label_roundtrip(self):
        for label_str, idx in LABEL_MAP.items():
            assert ID_TO_LABEL[idx] == label_str


class TestModelDefaults:
    """Verify model defaults have required keys."""

    def test_baseline_has_required_keys(self):
        required = {
            "embed_dim", "hidden_dim", "num_layers", "dropout",
            "batch_size", "max_seq_len", "learning_rate", "epochs", "grad_clip",
        }
        assert required <= set(BASELINE_DEFAULTS.keys())


class TestPaths:
    def test_assets_dir_exists(self):
        assert ASSETS_DIR.exists()

    def test_data_dir_is_assets(self):
        assert DATA_DIR == ASSETS_DIR

    def test_models_dir_is_assets(self):
        assert MODELS_DIR == ASSETS_DIR
