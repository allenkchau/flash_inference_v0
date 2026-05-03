import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.attention import Attention
from flash_inference.model.mlp import MLP

class TransformerBlock(nn.Module):
    """
    The repeating subunit of a full transformer model.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # we are using pre-layernorm for better stability
        x = self.layernorm(x)
        # apply attention
        x = self.attn(x)

        x = self.mlp(x)

        return x

