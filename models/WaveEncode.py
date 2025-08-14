import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Attention import Attention
from .MLP import MLP
from .Block import Block
from .Feature_Extractor import Feature_Extractor

class WaveEncode(nn.Module):
    def __init__(self, conv_configs:list[tuple[int, int, int]], nlayers:int, dim:int, nhead:int, ratio:float, dropout: float = 0.0, droppath: float = 0.0):
        super().__init__()
        self.fe = Feature_Extractor(conv_configs, dropout=dropout)
        self.embed = nn.Linear(conv_configs[-1][0], dim)
        self.blocks = nn.ModuleList([
            Block(dim=dim, nhead=nhead, ratio=ratio, droppath=droppath) for _ in range(nlayers)
        ])
        self.ln = nn.LayerNorm(dim)

    def feature_extractor(self, x:torch.Tensor, lengths:torch.Tensor):
        features, new_lengths = self.fe(x, lengths)
        features_transposed = features.permute(0, 2, 1)
        return features_transposed, new_lengths
    
    def encode(self, x:torch.Tensor, lengths:torch.Tensor|None):
        x = self.embed(x)
        max_len = x.shape[1]
        if lengths is not None:
            mask = (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None])
        else:
            mask = None
        for block in self.blocks:
            x, _ = block(x, mask)
        return self.ln(x)

        
    def forward(self, x:torch.Tensor, lengths:torch.Tensor):
        features_transposed, new_lengths = self.feature_extractor(x, lengths)
        max_len = features_transposed.shape[1]
        mask = (torch.arange(max_len)[None, :] < new_lengths[:, None])
        h = self.encode(features_transposed, mask)
        return h, new_lengths