import torch
import torch.nn as nn

from flash_inference.configs.model_config import ModelConfig

class Embeddings(nn.Module):
    """
    Every integer in the the original (batch_size, seq_len) grid is replaced by a full vector of size model_dim

    Use absolute positional embeddings instead of RoPE for simplicity.
    A position vector is added to the word embedding to produce the final input embedding.
    The position vector must have the same dimensions as the token embeddings since they are added together.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()

        # embeddings look up table
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim, device=config.device, dtype=config.dtype)

        # absolute positional embeddings for simplicty (for now)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.model_dim, device=config.device, dtype=config.dtype)


    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        seq_len = input_tokens.shape[1]

        # make sure seq_len doesn't exceed max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )

        # create position ids for the current sequence
        positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0)  # unsqueeze to add the batch dimension

        tok_embeddings = self.token_embeddings(input_tokens)  # shape: (batch_size, seq_len, model_dim)
        pos_embeddings = self.position_embeddings(positions)    # shape: (1, seq_len, model_dim)

        # add token and positional embeddings
        return tok_embeddings + pos_embeddings


        

