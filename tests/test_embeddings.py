import pytest
import torch

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.embeddings import Embeddings


def _make_config(
    *,
    model_dim: int = 16,
    max_seq_len: int = 32,
    vocab_size: int = 1000,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        model_dim=model_dim,
        num_heads=4,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        device=torch.device(device),
        dtype=dtype,
        activation="gelu",
    )


def test_embeddings_output_shape_dtype_and_device():
    emb = Embeddings(_make_config(model_dim=12))
    tokens = torch.randint(0, 1000, (3, 7), dtype=torch.long)

    out = emb(tokens)

    assert out.shape == (3, 7, 12)
    assert out.dtype == emb.token_embeddings.weight.dtype
    assert out.device == emb.token_embeddings.weight.device


def test_embeddings_equals_token_plus_position_lookup():
    emb = Embeddings(_make_config(model_dim=8))
    tokens = torch.randint(0, 1000, (2, 5), dtype=torch.long)

    out = emb(tokens)
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    expected = emb.token_embeddings(tokens) + emb.position_embeddings(positions)

    torch.testing.assert_close(out, expected)


def test_same_token_same_position_matches_across_batch():
    emb = Embeddings(_make_config(model_dim=10))
    repeated = torch.tensor([[42, 42, 42], [42, 42, 42]], dtype=torch.long)

    out = emb(repeated)
    torch.testing.assert_close(out[0], out[1])


def test_same_token_different_positions_produce_different_vectors():
    emb = Embeddings(_make_config(model_dim=6, vocab_size=50, max_seq_len=8))
    tokens = torch.tensor([[3, 3, 3, 3]], dtype=torch.long)

    with torch.no_grad():
        emb.token_embeddings.weight.zero_()
        emb.position_embeddings.weight.copy_(
            torch.arange(8 * 6, dtype=emb.position_embeddings.weight.dtype).reshape(8, 6)
        )

    out = emb(tokens)
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])
    assert not torch.allclose(out[:, 1, :], out[:, 2, :])


def test_embeddings_raises_when_sequence_exceeds_max_seq_len():
    emb = Embeddings(_make_config(max_seq_len=4))
    tokens = torch.randint(0, 1000, (2, 5), dtype=torch.long)

    with pytest.raises(IndexError):
        emb(tokens)
