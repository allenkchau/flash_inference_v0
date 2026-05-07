import torch

class PrefixKVCache:
    """
    This implementation of the cache removes external fragmentation and minimizes internal fragmentation (inspired by SGLang).

    We use a page table to keep track of which blocks in the table 
    """
    def __init__(self):
        pass
