import torch
from torch import nn
import torch.nn.functional as F

from .mask import GTNMMask


def nm_linear_forward(self: nn.Linear, input):
    weight = self.mask.apply_to_weight(self.weight)
    return F.linear(input, weight, self.bias)


def make_nm_linear_(model: nn.Linear, group_size, N, M, **mask_init_kwargs):
    n_group = (model.in_features * model.out_features) // M // group_size
    mask = GTNMMask(n_group, group_size, N, M, **mask_init_kwargs)
    setattr(model, "mask", mask)
    model.forward = nm_linear_forward.__get__(model)
    return model
