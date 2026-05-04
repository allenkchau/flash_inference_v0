import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.attention import MHAttention
from flash_inference.model.layernorm import LayerNorm
from flash_inference.model.mlp import MLP

class TransformerBlock(nn.Module):
    """
    The repeating subunit of a full transformer model.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MHAttention(config)
        self.mlp = MLP(config)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.attn(self.ln1(x))
        x = x + res
    
        res = x
        x = self.mlp(self.ln2(x))
        x = x + res

        return x

