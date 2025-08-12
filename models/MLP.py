import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, dim:int, ratio:float):
        super().__init__()

        self.hdim = int(ratio * dim)

        self.mlp_0 = nn.Parameter(
            torch.randn(
                self.hdim, dim
            )
        )

        self.mlp_1 = nn.Parameter(
            torch.randn(
                dim, self.hdim
            )
        )

    def forward(self, x):
        h = F.linear(x, self.mlp_0)
        h = F.gelu(h)
        h = F.linear(h, self.mlp_1)
        return h