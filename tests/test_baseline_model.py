"""Tests for baseline/model.py: BiLSTM + Attention architecture."""

import torch
import pytest

from diacritize.config import NUM_CLASSES, BASELINE_DEFAULTS
from diacritize.tokenizer import CharTokenizer
from diacritize.baseline.model import BahdanauAttention, BiLSTMDiacritizer


@pytest.fixture
def tok():
    return CharTokenizer()


@pytest.fixture
def model(tok):
    return BiLSTMDiacritizer(vocab_size=tok.vocab_size)


class TestBahdanauAttention:
    def test_output_shape(self):
        attn = BahdanauAttention(hidden_dim=512)
        x = torch.randn(2, 10, 512)  # batch=2, seq=10, hidden=512
        out = attn(x)
        assert out.shape == (2, 10, 512)

    def test_output_differs_from_input(self):
        """Attention should mix information across positions."""
        attn = BahdanauAttention(hidden_dim=64)
        x = torch.randn(1, 5, 64)
        out = attn(x)
        assert not torch.allclose(x, out)


class TestBiLSTMDiacritizer:
    def test_output_shape(self, model):
        input_ids = torch.randint(0, 50, (4, 20))  # batch=4, seq=20
        logits = model(input_ids)
        assert logits.shape == (4, 20, NUM_CLASSES)

    def test_output_shape_single_sample(self, model):
        input_ids = torch.randint(0, 50, (1, 10))
        logits = model(input_ids)
        assert logits.shape == (1, 10, NUM_CLASSES)

    def test_with_attention_mask(self, model):
        """Mask shouldn't crash the forward pass."""
        input_ids = torch.randint(0, 50, (2, 15))
        mask = torch.ones(2, 15)
        logits = model(input_ids, attention_mask=mask)
        assert logits.shape == (2, 15, NUM_CLASSES)

    def test_parameter_count_reasonable(self, model):
        """Should be roughly 4-5M params for default config."""
        params = model.count_parameters()
        assert 3_000_000 < params < 6_000_000, f"Got {params:,} params"

    def test_padding_idx_zero(self, model):
        """Embedding for PAD (index 0) should be all zeros."""
        pad_embedding = model.embedding.weight[0]
        assert torch.all(pad_embedding == 0)

    def test_gradients_flow(self, model):
        """Verify backprop works end-to-end."""
        input_ids = torch.randint(1, 50, (2, 10))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        # Check that embedding gradients exist
        assert model.embedding.weight.grad is not None

    def test_different_inputs_different_outputs(self, model):
        a = torch.tensor([[1, 2, 3, 4, 5]])
        b = torch.tensor([[5, 4, 3, 2, 1]])
        out_a = model(a)
        out_b = model(b)
        assert not torch.allclose(out_a, out_b)

    def test_batch_independence(self, model):
        """Each sample in a batch should be processed independently."""
        model.eval()
        x1 = torch.randint(1, 50, (1, 8))
        x2 = torch.randint(1, 50, (1, 8))
        # Run separately
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            # Run batched
            out_batch = model(torch.cat([x1, x2], dim=0))
        assert torch.allclose(out1, out_batch[:1], atol=1e-5)
        assert torch.allclose(out2, out_batch[1:], atol=1e-5)
