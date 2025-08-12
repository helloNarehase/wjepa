import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Attention import Attention
from .MLP import MLP
from .Block import Block

class Predictor(nn.Module):
    def __init__(self, in_dim:int, dim:int, nlayers:int, nhead:int, ratio:float, max_seq_len:int):
        super().__init__()

        self.embed = nn.Linear(
            in_dim, dim
        )
        self.blocks = nn.ModuleList([
            Block(dim=dim, nhead=nhead, ratio=ratio) for _ in range(nlayers)
        ])

        self.mask = nn.Parameter(
            torch.randn(
                1, 1, dim
            )
        )
        self.pos_embed = nn.Parameter(
            torch.randn(
                1, max_seq_len, dim
            )
        )


    def encode(self, x:torch.Tensor, mask:torch.Tensor|None):
        for block in self.blocks:
            x, _ = block(x, mask)
        return x

    def enc(self, x:torch.Tensor, context_indices:torch.Tensor, target_indices:torch.Tensor):
        B, N_ctx, _ = x.shape
        N_tgt = target_indices.shape[1]
        D = self.pos_embed.shape[-1]

        # 1. embed context tokens
        x = self.embed(x)

        # 2. create mask tokens
        mask_tokens = self.mask.expand(B, N_tgt, -1)

        # 3. get positional embeddings from indices
        pos_embed = self.pos_embed.expand(B, -1, -1)
        context_pos = torch.gather(pos_embed, 1, context_indices.unsqueeze(-1).expand(-1, -1, D))
        target_pos = torch.gather(pos_embed, 1, target_indices.unsqueeze(-1).expand(-1, -1, D))

        # 4. add positional embeddings
        x = x + context_pos
        mask_tokens = mask_tokens + target_pos

        # 5. concatenate context and target tokens
        predictor_input = torch.cat([x, mask_tokens], dim=1)

        # 6. pass through transformer blocks
        h = self.encode(predictor_input, mask=None)

        # 7. return predictions for target tokens
        pred = h[:, N_ctx:]
        return pred
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor|None):
        x = self.embed(x)
        h = self.encode(x, mask)
        return h
