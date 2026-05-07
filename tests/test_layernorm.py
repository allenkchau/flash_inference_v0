import torch
import torch.nn.functional as F

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.layernorm import LayerNorm


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
        vocab_size=1000,
        device=torch.device(device),
        dtype=dtype,
        activation="gelu",
    )


def test_layernorm_preserves_input_shape_dtype_device():
    x = torch.randn(2, 5, 16, dtype=torch.float32)
    ln = LayerNorm(_make_config(model_dim=x.shape[-1], dtype=x.dtype, device=x.device))

    out = ln(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_layernorm_has_trainable_gamma_and_beta():
    ln = LayerNorm(_make_config(model_dim=16))

    assert hasattr(ln, "gamma")
    assert hasattr(ln, "beta")
    assert isinstance(ln.gamma, torch.nn.Parameter)
    assert isinstance(ln.beta, torch.nn.Parameter)
    assert ln.gamma.shape == (16,)
    assert ln.beta.shape == (16,)
    assert ln.gamma.requires_grad
    assert ln.beta.requires_grad


def test_layernorm_normalizes_last_dimension_when_affine_is_identity():
    x = torch.randn(3, 7, 16)
    ln = LayerNorm(_make_config(model_dim=x.shape[-1]))
    with torch.no_grad():
        ln.gamma.fill_(1.0)
        ln.beta.zero_()

    out = ln(x)
    per_token_mean = out.mean(dim=-1)
    per_token_var = out.var(dim=-1, unbiased=False)

    assert torch.allclose(per_token_mean, torch.zeros_like(per_token_mean), atol=1e-4)
    assert torch.allclose(per_token_var, torch.ones_like(per_token_var), atol=2e-3)


def test_layernorm_matches_torch_reference():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    ln = LayerNorm(_make_config(model_dim=x.shape[-1], dtype=x.dtype))
    with torch.no_grad():
        ln.gamma.uniform_(0.8, 1.2)
        ln.beta.uniform_(-0.2, 0.2)

    out = ln(x)
    ref = F.layer_norm(x, normalized_shape=(x.shape[-1],), weight=ln.gamma, bias=ln.beta)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_layernorm_backpropagates_to_input_and_parameters():
    x = torch.randn(2, 3, 16, requires_grad=True)
    ln = LayerNorm(_make_config(model_dim=x.shape[-1]))

    out = ln(x)
    loss = out.square().mean()
    loss.backward()

    assert x.grad is not None
    assert ln.gamma.grad is not None
    assert ln.beta.grad is not None
