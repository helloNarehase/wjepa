import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Attention import Attention
from .MLP import MLP
from .Block import Block
from .Feature_Extractor import Feature_Extractor

class WaveEncode(nn.Module):
    def __init__(self, conv_configs:list[tuple[int, int, int]], nlayers:int, dim:int, nhead:int, ratio:float):
        super().__init__()
        self.feature_extractor = Feature_Extractor(conv_configs)
        self.blocks = nn.ModuleList([
            Block(dim=dim, nhead=nhead, ratio=ratio) for _ in range(nlayers)
        ])
        
    def forward(self, x:torch.Tensor, lengths:torch.Tensor):
        features, new_lengths = self.feature_extractor(x, lengths)
        features_transposed = features.permute(0, 2, 1)

        max_len = features_transposed.shape[1]
        mask = (torch.arange(max_len)[None, :] < new_lengths[:, None])
        h = features_transposed
        for block in self.blocks:
            h, _ = block(h, mask)
        return h, new_lengths