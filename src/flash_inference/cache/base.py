import torch
from flash_inference.configs.model_config import ModelConfig

class KVCache:
    """
    Let's think about the shape of our kv cache.
    We have a cache for each request we are handling (batch size).
    Each layer and attention head in the model has separate k and v vectors (num_layers and num_heads).
    We have both a K and a V vector which is the length of the sequence.
    Last dim is head_dim since each attn head only stores its slice.

    So the shape of one of our caches is (batch_size, num_layers, num_heads, seq_len, head_dim).
    """
    def __init__(self, config: ModelConfig, batch_size: int):
        # keep track of the length of our cache
        self.cur_len = 0

        # get some metrics from config
        self.num_layers = config.num_layers
        self.max_seq_len = config.max_seq_len

        # the base KV cache reserves max contiguous memory in advance
        self.k_cache = torch.empty(batch_size, config.num_layers, config.num_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)
        self.v_cache = torch.empty(batch_size, config.num_layers, config.num_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)
        

    # add new K and V vectors to the existing cache
    # we don't increment the length of the cache in this API because 
    def append(self, K_new: torch.Tensor, V_new: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # make sure layer_idx is in range
        assert 0 <= layer_idx < self.num_layers, "Layer idx for KV cache retrieval is not valid"

        # check that cur_len < max_seq_len
        assert self.cur_len < self.max_seq_len, "Out of bounds for KV cache write"

        # check that K_new and V_new shapes are valid
        assert self.cur_len < self.max_seq_len, "Out of bounds for KV cache write"

        self.k_cache[:, layer_idx, :, self.cur_len, :] = K_new
        self.v_cache[:, layer_idx, :, self.cur_len, :] = V_new
        self.cur_len += 1

    # retrieve a layer of the KV cache to use
    # by default we return all activate batch entries for that layer
    def retrieve(self, layer_idx: int) -> tuple(torch.Tensor):
        # make sure layer_idx is in range
        assert 0 <= layer_idx < self.num_layers, "Layer idx for KV cache retrieval is not valid"
        return self.k_cache[:, layer_idx, :, :self.cur_len, :], self.v_cache[:, layer_idx, :, :self.cur_len, :]

    # clean up the cache when we are done with a request
    def clear(self):
        # reset cache length
        self.cur_len = 0

