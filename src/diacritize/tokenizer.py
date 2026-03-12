"""Character-level tokenizer for Arabic text.

Converts Arabic characters to integer IDs for neural network input.
Only tokenizes base characters (diacritics are handled separately as labels).
"""

from diacritize.unicode_utils import strip_diacritics


# Special token IDs
PAD = 0   # Padding (for batching sequences of different lengths)
UNK = 1   # Unknown character (not in vocabulary)
BOS = 2   # Beginning of sequence
EOS = 3   # End of sequence

# Arabic letters (U+0621 – U+064A)
ARABIC_LETTERS = (
    "ءآأؤإئابةتثجحخدذرزسشصضطظعغ"
    "ـفقكلمنهوي"
)

# Common punctuation and whitespace the model will encounter
PUNCTUATION = " .,،؛:؟!()-\"'\n"

# Special token strings (for building vocabulary)
_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


class CharTokenizer:
    """Character-level tokenizer for Arabic diacritization.

    Maps each base character to an integer ID. Diacritics are NOT tokenized
    here — they're handled by extract_diacritics() as labels.

    Attributes:
        char_to_id: Dict mapping characters to integer IDs.
        id_to_char: Dict mapping integer IDs back to characters.
        vocab_size: Total number of tokens in vocabulary.
    """

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {}

        # Special tokens at indices 0-3
        for idx, token in enumerate(_SPECIAL_TOKENS):
            self.char_to_id[token] = idx

        # Arabic letters from index 4 onwards
        offset = len(_SPECIAL_TOKENS)
        for i, char in enumerate(ARABIC_LETTERS):
            self.char_to_id[char] = offset + i

        # Punctuation after Arabic letters
        offset += len(ARABIC_LETTERS)
        for i, char in enumerate(PUNCTUATION):
            self.char_to_id[char] = offset + i

        # Reverse mapping and vocab size
        self.id_to_char: dict[int, str] = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size: int = len(self.char_to_id)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Convert text to a list of integer IDs.

        Strips diacritics first (we only tokenize base characters).
        Unknown characters map to UNK.

        Args:
            text: Arabic text (may contain diacritics, they'll be stripped).
            add_special: If True, wrap with BOS and EOS tokens.

        Returns:
            List of integer token IDs.
        """

        text = strip_diacritics(text)
        ids = [self.char_to_id.get(char, UNK) for char in text]
        if add_special:
            ids = [BOS] + ids + [EOS]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert integer IDs back to text.

        Args:
            ids: List of integer token IDs.
            skip_special: If True, skip PAD/BOS/EOS tokens in output.

        Returns:
            Decoded string.
        """
        chars = []
        for id in ids:
            if skip_special and id in {PAD, BOS, EOS}:
                continue
            if id == UNK:
                chars.append("□")  # Single placeholder for unknown chars
            else:
                chars.append(self.id_to_char.get(id, "□"))
        return "".join(chars)
