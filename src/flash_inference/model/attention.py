import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig

class MHAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # matrices
        self.Wq = nn.Linear(in_features=config.model_dim, out_features=config.model_dim, device=config.device, dtype=config.dtype)
        self.Wk = nn.Linear(in_features=config.model_dim, out_features=config.model_dim, device=config.device, dtype=config.dtype)
        self.Wv = nn.Linear(in_features=config.model_dim, out_features=config.model_dim, device=config.device, dtype=config.dtype)

        

    def forward(self, x: torch.tensor) -> torch.tensor:
        # apply transformations to get Q, K, V
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # split into attention heads

        # 

        return 


# other popular forms of attention are implemented here


