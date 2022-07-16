import torch
import numpy as np
import random


def initialize_seeds(seed):
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# credit to https://github.com/alinlab/LfF/blob/master/util.py
class MultiDimAverageMeter:
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_mean(self, EPS=0):
        return (self.cum / (self.cnt + EPS)).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()
