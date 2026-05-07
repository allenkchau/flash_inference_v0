import torch

class PagedKVCache:
    """
    This implementation of the cache removes external fragmentation and minimizes internal fragmentation (inspired by vLLM).

    We use a page table to keep track of which blocks in the table 
    """
    def __init__(self, block_size: int):
        # keep track of the length of our cache
        self.cur_len = 0

        self.block_size = block_size
        self.block_table = 


    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        pass

    # retrieve a layer of the KV cache to use
    # we need to go to our block table and see what blocks our layer maps to
    def retrieve(self, layer_idx: int) -> tuple(torch.Tensor, torch.Tensor):
        # make sure layer_idx is in range
        assert 0 <= layer_idx < self.num_layers, "Layer idx for KV cache retrieval is not valid"
        return (self.k_cache[:, layer_idx, :, :self.cur_len, :], self.v_cache[:, layer_idx, :, :self.cur_len, :])

    # clean up the cache when we are done with a request
    def clear(self):
        # reset cache length
        self.cur_len = 0

        # clear our block table
        


