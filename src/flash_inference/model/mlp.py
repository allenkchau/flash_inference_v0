import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import ACTIVATIONS

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # two linear transformations
        self.W1 = nn.Linear(in_features=config.model_dim, out_features=config.mlp_hidden_size, device=config.device, dtype=config.dtype)
        self.W2 = nn.Linear(in_features=config.mlp_hidden_size, out_features=config.model_dim, device=config.device, dtype=config.dtype)

        # non-linear activation between the linear transformations
        self.activation = ACTIVATIONS[config.activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.activation(self.W1(x)))
