import torch
from flash_inference.configs.model_config import ModelConfig

class KVCache:
    """
    Let's think about the shape of our kv cache.
    We have a cache for each request we are handling (batch size).
    Each layer in the model has separate k and v vectors (num_layers).
    We have both a K and a V vector which is the length of the sequence.

    So the shape of one of our caches is (batch_size, num_layers, num_heads, seq_len, model_dim).
    """
    def __init__(self, config: ModelConfig,):
        # keep track of the length of our cache
        self.cur_len = 0

        # the base KV cache reserves contiguous memory in advance
        self.k_cache = torch.empty(config.num_layers, config.num_heads, seq_len, config.model_dim)
        self.v_cache = 
        

    # add new K and V vectors to the existing cache
    def append(K_new: torch.tensor, V_new: torch.tensor) -> torch.tensor:
        for

    # clean up the cache when we are done with a request
    def clear(self):
