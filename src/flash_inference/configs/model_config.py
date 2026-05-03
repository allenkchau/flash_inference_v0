import torch
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """
    Config to use when initializing a transformer model.
    """
    num_layers: int
    model_dim: int
    num_heads: int
    max_seq_len: int

    device: torch.device
    dtype: torch.dtype

    # activation for MLP
    activation: Literal["relu", "gelu", "silu"]

    @property
    def head_dim(self) -> int:
        assert self.model_dim % self.num_heads == 0
        return self.model_dim // self.num_heads

    @property
    def mlp_hidden_size(self) -> int:
        return self.model_dim * 4

    






