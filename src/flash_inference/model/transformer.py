import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.block import TransformerBlock
from flash_inference.model.embeddings import Embeddings
from flash_inference.model.layernorm import LayerNorm

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # we need nn.ModuleList here instead of just a normal Python list
        # if we use a normal list, the layers' parameters won't be correctly registered and trained
        self.layers: list[TransformerBlock] = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.embeddings = Embeddings(config)

        # final layer norm
        # after the many residual additions in the blocks, before we turn hidden states -> logits, we want to make sure activations are well-scaled
        self.ln = LayerNorm(config)

        # final linear layer: converts model_dim -> vocab size
        # check if we use weight tying
        # we don't need a bias term because usually there is not such term in the output head
        self.output = nn.Linear(
            in_features=config.model_dim,
            out_features=config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids has shape: (batch_size, seq_len)
        # first we embed the input ids to get shape: (batch_size, seq_len, model_dim)
        x = self.embeddings(input_ids)

        # run through all the layers
        for layer in self.layers:
            x = layer(x)

        x = self.ln(x)

        # just some notes here
        # at this point x has shape: batch_size, seq_len, model_dim
        # it's helpful to think of each slice x[b, t, :] as the model's contextualized represenation of token t after seeing all tokens up to t

        # output head to generate
        logits = self.output(x)
        return logits
