import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Attention import Attention
from .MLP import MLP

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):
    def __init__(self, dim:int, nhead:int, ratio:float, droppath: float = 0.0):
        super().__init__()

        self.hdim = int(ratio * dim)

        self.attn = Attention(dim=dim, nheads=nhead)
        self.mlp  = MLP(dim=dim, ratio=ratio)

        self.post_ln      = nn.LayerNorm(dim, eps=1e-8)
        self.attention_ln = nn.LayerNorm(dim, eps=1e-8)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

    def forward(self, x, mask):
        h, du = self.attn(self.post_ln(x), mask)
        h = x + self.drop_path(h)
        h = h + self.drop_path(self.mlp(self.attention_ln(h)))
        return h, du