import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Attention import Attention
from .MLP import MLP

class Block(nn.Module):
    def __init__(self, dim:int, nhead:int, ratio:float):
        super().__init__()

        self.hdim = int(ratio * dim)

        self.attn = Attention(dim=dim, nheads=nhead)
        self.mlp  = MLP(dim=dim, ratio=ratio)

        self.post_ln      = nn.LayerNorm(dim, eps=1e-8)
        self.attention_ln = nn.LayerNorm(dim, eps=1e-8)

    def forward(self, x, mask):
        h, du = self.attn(self.post_ln(x), mask)
        h += x
        h = self.mlp(self.attention_ln(h)) + h
        return h, du