import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Attention import Attention
from .MLP import MLP
from .Block import Block
from .Feature_Extractor import Feature_Extractor

class Blocks(nn.Module):
    def __init__(self, last_conv:int, nlayers:int, dim:int, nhead:int, ratio:float, dropout: float = 0.0, droppath: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(last_conv, dim)
        self.blocks = nn.ModuleList([
            Block(dim=dim, nhead=nhead, ratio=ratio, droppath=droppath) for _ in range(nlayers)
        ])
        self.ln = nn.LayerNorm(dim)

    def forward(self, x:torch.Tensor, lengths:torch.Tensor|None):
        x = self.embed(x)
        max_len = x.shape[1]
        if lengths is not None:
            mask = (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None])
        else:
            mask = None
        for block in self.blocks:
            x, _ = block(x, mask)
        return self.ln(x)

class WaveEncode(nn.Module):
    def __init__(self, conv_configs:list[tuple[int, int, int]], nlayers:int, dim:int, nhead:int, ratio:float, dropout: float = 0.0, droppath: float = 0.0):
        super().__init__()
        self.fe = Feature_Extractor(conv_configs, dropout=dropout)
        self.blocks = Blocks(
            last_conv=conv_configs[-1][0],
            nlayers=nlayers,
            dim=dim,
            nhead=nhead,
            ratio=ratio,
            dropout=dropout,
            droppath=droppath
        )

    def feature_extractor(self, x:torch.Tensor, lengths:torch.Tensor):
        features, new_lengths = self.fe(x, lengths)
        # features_transposed = features
        return features, new_lengths
    
    def encode(self, x:torch.Tensor, lengths:torch.Tensor|None):
        return self.blocks(x, lengths)

    def forward(self, x:torch.Tensor, lengths:torch.Tensor):
        features, new_lengths = self.feature_extractor(x, lengths)
        features_transposed = features.permute(0, 2, 1)
        h = self.encode(features_transposed, new_lengths)
        return h, new_lengths