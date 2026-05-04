import importlib

import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig


def _make_config(
    model_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        model_dim=model_dim,
        num_heads=4,
        max_seq_len=32,
        device=torch.device(device),
        dtype=dtype,
        activation="gelu",
    )


def _load_block_class():
    """
    Keep block tests runnable while Attention is still under development.
    If `Attention` is not exported yet, inject a temporary identity fallback.
    """
    attention_module = importlib.import_module("flash_inference.model.attention")

    if not hasattr(attention_module, "Attention"):
        class _FallbackAttention(nn.Module):
            def __init__(self, config: ModelConfig):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        attention_module.Attention = _FallbackAttention

    block_module = importlib.import_module("flash_inference.model.block")
    block_module = importlib.reload(block_module)
    return block_module.TransformerBlock


def test_transformer_block_preserves_shape_dtype_and_device():
    TransformerBlock = _load_block_class()
    x = torch.randn(2, 5, 16, dtype=torch.float32)
    block = TransformerBlock(_make_config(model_dim=x.shape[-1], dtype=x.dtype, device=x.device))

    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_transformer_block_has_expected_submodules():
    TransformerBlock = _load_block_class()
    block = TransformerBlock(_make_config(model_dim=16))

    assert hasattr(block, "attn")
    assert hasattr(block, "mlp")
    assert hasattr(block, "ln1")
    assert hasattr(block, "ln2")
    assert isinstance(block.attn, nn.Module)
    assert isinstance(block.mlp, nn.Module)
    assert isinstance(block.ln1, nn.Module)
    assert isinstance(block.ln2, nn.Module)


def test_transformer_block_calls_attention_once_mlp_once_and_uses_ln2():
    TransformerBlock = _load_block_class()
    block = TransformerBlock(_make_config(model_dim=16))
    x = torch.randn(2, 3, 16)

    class _CountingModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return y

    attn = _CountingModule()
    mlp = _CountingModule()
    ln1 = _CountingModule()
    ln2 = _CountingModule()

    block.attn = attn
    block.mlp = mlp
    block.ln1 = ln1
    block.ln2 = ln2

    _ = block(x)

    assert attn.calls == 1, "TransformerBlock should call attention once per forward pass."
    assert mlp.calls == 1, "TransformerBlock should call MLP once per forward pass."
    assert ln1.calls == 1, "TransformerBlock should use first layernorm once per forward pass."
    assert ln2.calls == 1, "TransformerBlock should use second layernorm once per forward pass."

