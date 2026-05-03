import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        # matrices
        self.Wq = nn.Linear()
        self.Wk = nn.Linear()

    def forward(x: torch.Tensor) -> torch.Tensor:
        # apply transformations to get Q, K, V
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        # split into attention heads
