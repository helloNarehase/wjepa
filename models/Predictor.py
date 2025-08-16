import torch
from torch import einsum, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .Attention import Attention
from .MLP import MLP
from .Block import Block

from typing import List, Tuple, Optional, Callable


def create_span_targets(features: torch.Tensor, target_mask: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Creates targets for span-level tasks from the original features.
    
    Args:
        features: (B, n_mels, T) - Original, unmasked features.
        target_mask: (B, N, S) - Span indices to extract as targets.
        
    Returns:
        Target features for reconstruction (B, N, S, n_mels) or None.
    """
    if target_mask is None or target_mask.numel() == 0:
        return None
    
    B, N, S = target_mask.shape
    n_mels = features.shape[1]
    
    span_targets = torch.zeros(B, N, S, n_mels, dtype=features.dtype, device=features.device)
    
    for b in range(B):
        for n in range(N):
            for s in range(S):
                frame_idx = target_mask[b, n, s].item()
                if frame_idx != -1 and frame_idx < features.shape[2]:
                    span_targets[b, n, s] = features[b, :, frame_idx]
    
    return span_targets
    
def apply_masks(features: torch.Tensor, full_masks: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies masks by dropping the selected feature frames.
    
    Args:
        features: (B, n_mels, T) - Original mel spectrogram features.
        full_masks: A list of B tensors, each containing frame indices to drop.
        
    Returns:
        - dropped_padded_features: (B, n_mels, T_unmasked) Features after dropping and padding.
        - unmasked_lengths: (B,) Tensor with the lengths of each sequence before padding.
    """
    B, D, T = features.shape
    unmasked_features_list = []
    unmasked_lengths = []
    
    for i in range(B):
        drop_indices = full_masks[i].to(features.device)
        
        # Create a boolean mask of frames to KEEP
        keep_mask = torch.ones(T, dtype=torch.bool, device=features.device)
        if drop_indices.numel() > 0:
            keep_mask[drop_indices] = False
            
        # Select frames to keep: features[i] is (n_mels, T)
        unmasked_sample = features[i][:, keep_mask]
        unmasked_features_list.append(unmasked_sample)
        unmasked_lengths.append(unmasked_sample.shape[1])
        
    # Pad the sequences to the max length in the batch
    # pad_sequence expects (T, *), so we need to transpose, pad, then transpose back
    unmasked_features_transposed = [f.transpose(0, 1) for f in unmasked_features_list]
    padded_transposed = pad_sequence(unmasked_features_transposed, batch_first=True, padding_value=0.0)
    
    # Transpose back to (B, n_mels, T_padded)
    dropped_padded_features = padded_transposed.transpose(1, 2)
    
    return dropped_padded_features, torch.tensor(unmasked_lengths, dtype=torch.long, device=features.device)

class Predictor(nn.Module):
    def __init__(self, in_dim:int, dim:int, nlayers:int, nhead:int, ratio:float, max_seq_len:int, droppath: float = 0.0):
        super().__init__()

        self.embed = nn.Linear(
            in_dim, dim
        )
        self.blocks = nn.ModuleList([
            Block(dim=dim, nhead=nhead, ratio=ratio, droppath=droppath) for _ in range(nlayers)
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
        self.ln = nn.LayerNorm(dim)

    def forward(self, ctx_feature:torch.Tensor, length:int, ctx_mask:torch.Tensor, tgt_mask:torch.Tensor):
        B, _, _ = ctx_feature.shape
        _, M, N_tgt = tgt_mask.shape
        N_ctx_total = length

        # ctx
        pos_embed = self.pos_embed.expand(B, -1, -1)[:, :N_ctx_total, :]
        pos_embed, _lengths = apply_masks(pos_embed.permute(0, 2, 1), ctx_mask)
        ctx_feature += pos_embed
        
        ctx_feature = torch.repeat_interleave(
            ctx_feature.permute(0, 2, 1)[:, None, :, :],
            repeats=M,
            dim=1
        )
        ctx_feature = ctx_feature.reshape(B * M, _lengths.max().item(), -1)

        # tgt
        pos_embed_base_repeated = self.pos_embed.expand(B, -1, -1)[:, :N_ctx_total, :].repeat_interleave(M, dim=0)
        tgt_mask_reshaped = tgt_mask.reshape(B * M, 1, N_tgt)
        
        tgt_pos_embed = create_span_targets(
            pos_embed_base_repeated.permute(0, 2, 1), tgt_mask_reshaped
        )
        tgt_pos_embed = tgt_pos_embed.squeeze(1)

        tgt_masks = self.mask.expand(B * M, N_tgt, -1)
        tgt_masks = tgt_masks + tgt_pos_embed
        
        # attention-mask (padding mask)
        N_ctx = _lengths.max()
        attention_mask = (torch.arange(N_ctx + N_tgt, device=_lengths.device)[None, :] < _lengths[:, None])
        attention_mask = torch.repeat_interleave(
            attention_mask,
            repeats=M, 
            dim=0
        )
        attention_mask[..., -N_tgt:] = True

        # forward
        h = torch.cat([self.embed(ctx_feature), tgt_masks], dim=1)
        for block in self.blocks:
            h, _ = block(h, attention_mask)
        h = self.ln(h)

        return h[:, -N_tgt:, :]