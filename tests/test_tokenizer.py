"""Tests for tokenizer.py: character-level Arabic tokenizer."""

import pytest

from diacritize.tokenizer import (
    CharTokenizer,
    PAD,
    UNK,
    BOS,
    EOS,
    ARABIC_LETTERS,
)


@pytest.fixture
def tok():
    return CharTokenizer()


class TestVocabulary:
    def test_special_token_ids(self):
        assert PAD == 0
        assert UNK == 1
        assert BOS == 2
        assert EOS == 3

    def test_vocab_size_at_least_special_plus_letters(self, tok):
        # 4 special + 39 Arabic letters + punctuation
        assert tok.vocab_size >= 4 + len(ARABIC_LETTERS)

    def test_all_arabic_letters_in_vocab(self, tok):
        for letter in ARABIC_LETTERS:
            assert letter in tok.char_to_id, f"{letter} missing from vocab"

    def test_no_duplicate_ids(self, tok):
        ids = list(tok.char_to_id.values())
        assert len(ids) == len(set(ids))

    def test_id_to_char_is_reverse(self, tok):
        for char, idx in tok.char_to_id.items():
            assert tok.id_to_char[idx] == char

    def test_space_in_vocab(self, tok):
        assert " " in tok.char_to_id


class TestEncode:
    def test_simple_word(self, tok):
        ids = tok.encode("بسم", add_special=False)
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)

    def test_with_special_tokens(self, tok):
        ids = tok.encode("بسم", add_special=True)
        assert ids[0] == BOS
        assert ids[-1] == EOS
        assert len(ids) == 5  # BOS + 3 chars + EOS

    def test_strips_diacritics_before_encoding(self, tok):
        """مُحَمَّدٌ should encode the same as محمد."""
        ids_diacritized = tok.encode("مُحَمَّدٌ", add_special=False)
        ids_plain = tok.encode("محمد", add_special=False)
        assert ids_diacritized == ids_plain

    def test_unknown_char_maps_to_unk(self, tok):
        ids = tok.encode("@", add_special=False)
        assert ids == [UNK]

    def test_empty_string(self, tok):
        ids = tok.encode("", add_special=False)
        assert ids == []

    def test_empty_string_with_special(self, tok):
        ids = tok.encode("", add_special=True)
        assert ids == [BOS, EOS]

    def test_space_encoded(self, tok):
        ids = tok.encode("ب م", add_special=False)
        assert len(ids) == 3  # ba + space + mim


class TestDecode:
    def test_simple_roundtrip(self, tok):
        text = "بسم الله"
        ids = tok.encode(text, add_special=False)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_roundtrip_with_special_tokens(self, tok):
        text = "بسم"
        ids = tok.encode(text, add_special=True)
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == text

    def test_skip_special_removes_bos_eos(self, tok):
        decoded = tok.decode([BOS, EOS], skip_special=True)
        assert decoded == ""

    def test_keep_special_tokens(self, tok):
        ids = [BOS, tok.char_to_id["ب"], EOS]
        decoded = tok.decode(ids, skip_special=False)
        assert "<BOS>" in decoded
        assert "<EOS>" in decoded

    def test_pad_skipped(self, tok):
        ids = [tok.char_to_id["ب"], PAD, PAD]
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == "ب"
