import torch

from torch.nn.modules.loss import _WeightedLoss

import torch.nn.functional as F

###

class SimSiamLoss(_WeightedLoss):
    def __init__(self,
                 weight: torch.Tensor = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 ):
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self,
                p1: torch.Tensor,
                p2: torch.Tensor,
                z1: torch.Tensor,
                z2: torch.Tensor,
                ):
        # p1 = p1.unsqueeze(dim=0)
        # p2 = p2.unsqueeze(dim=0)
        # z1 = z1.unsqueeze(dim=0)
        # z2 = z2.unsqueeze(dim=0)

        z1 = z1.detach()
        z2 = z2.detach()

        loss = self._D(p1,z2).mean()*0.5 + self._D(p2,z1).mean()*0.5
        return loss

    def _D(self,
           p: torch.Tensor, 
           z: torch.Tensor,
           dim: int = 1,
           epsilon: float = 1e-8,
           ):
        return - F.cosine_similarity(p, z, dim=dim, eps=epsilon)