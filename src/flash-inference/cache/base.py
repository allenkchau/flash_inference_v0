import torch

class KVCache:
    def __init__(self):
        # keep track of the length of our cache
        self.cur_len = 0

        

    # add new K and V vectors to the existing cache
    def append(K_new: torch.Tensor, V_new: torch.Tensor) -> torch.Tensor:
        pass
