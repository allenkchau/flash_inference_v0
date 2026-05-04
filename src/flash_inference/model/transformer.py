import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers: list[TransformerBlock] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x)
        logits = 
        return logits
