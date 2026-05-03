import torch
import torch.nn as nn

from model.attention import Attention

class TransformerBlock(nn.Module):
    """
    The repeating subunit of a full transformer model.
    """
    def __init__(self):
        self.attn = Attention

    def forward(x: torch.Tensor) -> torch.Tensor:
        # apply attention
        x = self.attn(x)

