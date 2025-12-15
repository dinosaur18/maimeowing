import torch
from torch import nn


generator = None
gumbel_seed = 123


def gumbel_soft_topk(
    logits: torch.Tensor,
    k: int,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-12,
    p: float = 3,       # consider to use
) -> torch.Tensor:
    global generator
    # if generator is None:
        # generator = torch.Generator(logits.device)
        # offset seed
        # generator.manual_seed(gumbel_seed + torch.distributed.get_rank())
    
    g = torch.empty_like(logits).exponential_(generator=generator).log_().neg_()
    logits = logits + g
    
    eps = torch.finfo(logits.dtype).tiny
    eps = torch.tensor([eps], dtype=logits.dtype, device=logits.device)

    # continuous top k
    khot = torch.zeros_like(logits)
    onehot_approx = torch.zeros_like(logits)
    for i in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, eps)
        logits = logits + torch.log(khot_mask)
        onehot_approx = nn.functional.softmax(logits / tau, dim=1)
        khot = khot + onehot_approx

    if hard:
        # will do straight through estimation if training
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot

    return res


class GTNMMask(nn.Module):

    logit_scaling: float = 1.0
    tau: float = 1.0
    
    def __init__(
        self,
        n_group: int,
        group_size: int,
        N: int,
        M: int,
        hard: bool = False,
        init_std: float = 1e-2,
        eps: float = 1e-10,
        logit_scaling: float = 1.0,
        tau: float = 1.0
    ):
        super().__init__()

        assert group_size == 1, "group_size != 1 is not supported"
        
        self.n_group = n_group
        self.group_size = group_size
        self.N = N
        self.M = M
        self.hard = hard
        self.init_std = init_std
        self.eps = eps
        
        GTNMMask.logit_scaling = logit_scaling
        GTNMMask.tau = tau
        
        self.logits = nn.Parameter(torch.empty(n_group, M), requires_grad=True)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.logits.data.normal_(std=self.init_std)
    
    def apply_to_weight(self, weight):
        w = weight.view(-1, self.n_group, self.M) * self()
        w = w.view_as(weight)
        return w
    
    def forward(self):
        if self.training:
            # orig_dtype = self.logits.dtype
            # logits = self.logits.float()
            logits = self.logits * self.logit_scaling
            mask = gumbel_soft_topk(logits, self.N, self.tau, self.hard, self.eps)
            # mask = mask.to(orig_dtype)
        else:
            mask = torch.zeros_like(self.logits)
            _, indices = torch.topk(self.logits, self.N)
            mask = mask.scatter_(-1, indices, 1.0)
        
        return mask
