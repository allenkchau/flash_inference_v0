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

        # the base KV cache reserves max contiguous memory in advance
        self.k_cache = torch.empty(batch_size, config.num_layers, config.num_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)
        self.v_cache = torch.empty(batch_size, config.num_layers, config.num_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)
        

    # write one layer's KV for the current time step
    # external caller advances cur_len once all layers are written for that token step
    def append(self, K_new: torch.Tensor, V_new: torch.Tensor, layer_idx: int):

        expected = (
            self.k_cache.shape[0],  # B
            self.k_cache.shape[2],  # H
            self.k_cache.shape[4],  # D
        )

        # make sure layer_idx is in range
        assert 0 <= layer_idx < self.k_cache.shape[1], "Layer idx for KV cache retrieval is not valid"

        # check that cur_len < max_seq_len
        assert self.cur_len < self.k_cache.shape[3], "KV cache is full"
        assert K_new.dtype == self.k_cache.dtype
        assert K_new.device == self.k_cache.device
        assert V_new.dtype == self.v_cache.dtype
        assert V_new.device == self.v_cache.device

        # check that K_new and V_new shapes are valid
        assert K_new.shape == expected, f"K_new shape {tuple(K_new.shape)} != {expected}"
        assert V_new.shape == expected, f"V_new shape {tuple(V_new.shape)} != {expected}"

        self.k_cache[:, layer_idx, :, self.cur_len, :] = K_new
        self.v_cache[:, layer_idx, :, self.cur_len, :] = V_new

    # retrieve a layer of the KV cache to use
    # by default we return all activate batch entries for that layer
    def retrieve(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # make sure layer_idx is in range
        assert 0 <= layer_idx < self.k_cache.shape[1], "Layer idx for KV cache retrieval is not valid"
        return (self.k_cache[:, layer_idx, :, :self.cur_len, :], self.v_cache[:, layer_idx, :, :self.cur_len, :])

    # clean up the cache when we are done with a request
    def clear(self):
        # reset cache length
        self.cur_len = 0

