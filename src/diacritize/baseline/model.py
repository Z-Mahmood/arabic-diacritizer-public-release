"""BiLSTM + Bahdanau Attention model for Arabic diacritization.

Architecture:
    Input IDs → Embedding(128) → BiLSTM(256×2, 3 layers) → Attention(512) → Linear(15)

The BiLSTM reads the character sequence in both directions. The attention
mechanism lets each position attend to the full sequence when making its
prediction. The final linear layer outputs 15 class logits (one per diacritic).

~4.5M parameters. Trains in ~3.5 hrs on RTX 5080.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diacritize.config import BASELINE_DEFAULTS, ID_TO_LABEL, MODELS_DIR, NUM_CLASSES
from diacritize.tokenizer import CharTokenizer
from diacritize.unicode_utils import apply_diacritics, strip_diacritics


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention over BiLSTM hidden states.

    For each position t, computes a weighted sum of ALL hidden states,
    where weights are learned based on how relevant each state is to
    position t. This lets the model "look at" the full sequence context
    when predicting each character's diacritic.

    Args:
        hidden_dim: Size of BiLSTM hidden state (×2 for bidirectional).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute self-attention over the sequence.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) from BiLSTM.

        Returns:
            context: (batch, seq_len, hidden_dim) — attention-weighted output.
                     Same shape as input, but each position now contains
                     information from the full sequence.
        """

        query = self.W(hidden_states)
        query = query.unsqueeze(2)

        keys = self.U(hidden_states)
        keys = keys.unsqueeze(1)

        scores = self.v(torch.tanh(query + keys)).squeeze(-1)

        weights = torch.softmax(scores, dim=-1)

        context = torch.bmm(weights, hidden_states)

        return context

class BiLSTMDiacritizer(nn.Module):
    """Full BiLSTM + Attention model for diacritization.

    Architecture:
        Embedding → Dropout → BiLSTM → Attention → Linear → Logits

    Args:
        vocab_size: Number of tokens in character vocabulary.
        embed_dim: Embedding dimension (default 128).
        hidden_dim: LSTM hidden size per direction (default 256).
                    BiLSTM output is hidden_dim × 2.
        num_layers: Number of stacked LSTM layers (default 3).
        num_classes: Number of diacritic classes (default 15).
        dropout: Dropout rate (default 0.3).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = BASELINE_DEFAULTS["embed_dim"],
        hidden_dim: int = BASELINE_DEFAULTS["hidden_dim"],
        num_layers: int = BASELINE_DEFAULTS["num_layers"],
        num_classes: int = NUM_CLASSES,
        dropout: float = BASELINE_DEFAULTS["dropout"],
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.attention = BahdanauAttention(hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) integer token IDs.
            attention_mask: (batch, seq_len) optional, 1 for real tokens.
                           Not used in computation but kept for API consistency.

        Returns:
            logits: (batch, seq_len, num_classes) — raw scores for each diacritic.
        """
        x = self.embedding(input_ids)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        logits = self.classifier(x)

        return logits


    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Inference API ─────────────────────────────────────────────────

    def diacritize(self, text: str) -> str:
        """Diacritize a single Arabic text string.

        Strips existing diacritics, runs model inference, and re-applies
        predicted diacritics. The model must have a `tokenizer` attribute
        (set by `from_pretrained` or manually).
        """
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise RuntimeError(
                "No tokenizer attached. Use BiLSTMDiacritizer.from_pretrained() "
                "or set model.tokenizer = CharTokenizer() before calling diacritize()."
            )
        device = next(self.parameters()).device
        stripped = strip_diacritics(text)
        input_ids = self.tokenizer.encode(stripped, add_special=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self(input_tensor)
        pred_ids = logits.argmax(dim=-1)[0].tolist()
        labels = [ID_TO_LABEL[i] for i in pred_ids[: len(stripped)]]
        return apply_diacritics(stripped, labels)

    @classmethod
    def from_pretrained(cls, path: str | None = None) -> BiLSTMDiacritizer:
        """Load a trained BiLSTM model with tokenizer, ready for inference.

        Args:
            path: Path to the .pt checkpoint file.
                  Defaults to models/bilstm_best.pt.
        """
        if path is None:
            path = str(MODELS_DIR / "bilstm_best.pt")
        tokenizer = CharTokenizer()
        model = cls(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        model.set_to_inference_mode()
        model.tokenizer = tokenizer
        return model

    def set_to_inference_mode(self) -> None:
        """Set model to inference mode."""
        super().eval()

    def save_pretrained(self, path: str) -> None:
        """Save model weights to a .pt file."""
        torch.save(self.state_dict(), path)
