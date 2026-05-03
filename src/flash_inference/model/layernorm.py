import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig

class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = []

    def forward(self, ):
        return x
