import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig

class LayerNorm(nn.Module):
    """
    The purpose of layernorm is to stabilize training of the transformer.
    It normalizes the inputs across the feature-dimension for each token separately.
    Without it, we can get vanishing or exploding gradients.

    Modern LLMs implement pre-layernorm (layernorm before attn or FFN block) because it is more stable than the post-layernorm alternate.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()

        # learnable shift and scale parameters (use nn.Parameter to register them as actual parameters with gradients)
        self.gamma = nn.Parameter(torch.ones(config.model_dim, device=config.device, dtype=config.dtype))
        self.beta = nn.Parameter(torch.zeros(config.model_dim, device=config.device, dtype=config.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x: (batch_size, seq_len, model_dim)
        # the "features" of each token is basically model_dim
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, correction=0)  # correction arg tells torch.var to compute population variance and divide by N instead of N - 1 (sample variance)

        x = (x - mean) / torch.sqrt(var + 1e-5)     # small epsilon to prevent division by 0

        # apply shift and scale params
        x = self.gamma * x + self.beta

        return x
