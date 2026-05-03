import torch

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.mlp import MLP


def _make_config(
    model_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    activation: str = "gelu",
) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        model_dim=model_dim,
        num_heads=4,
        device=torch.device(device),
        dtype=dtype,
        activation=activation,
    )


def test_mlp_preserves_input_shape():
    x = torch.randn(2, 4, 16)
    mlp = MLP(_make_config(model_dim=x.shape[-1]))

    out = mlp(x)
    assert out.shape == x.shape


def test_mlp_preserves_dtype_and_device():
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    mlp = MLP(_make_config(model_dim=x.shape[-1], dtype=x.dtype, device=x.device))

    out = mlp(x)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_mlp_allows_backprop_to_input():
    x = torch.randn(2, 5, 12, requires_grad=True)
    mlp = MLP(_make_config(model_dim=x.shape[-1]))

    out = mlp(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_mlp_has_trainable_parameters():
    mlp = MLP(_make_config(model_dim=16))
    parameter_count = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    assert parameter_count > 0, "MLP should contain trainable parameters."


def test_mlp_is_not_identity_mapping():
    x = torch.randn(2, 4, 16)
    mlp = MLP(_make_config(model_dim=x.shape[-1]))

    out = mlp(x)
    assert out is not x, "MLP should return a new tensor, not input reference."
