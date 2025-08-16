import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample.
    - 입력 텐서 x에 대해 drop_prob 확률로 path를 drop합니다. (0으로 만듭니다)
    - 논문: "Deep Networks with Stochastic Depth"
    - PyTorch/Vision/timm 공식 구현을 참고했습니다.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # (batch, 1, 1) 또는 (batch, 1, 1, 1) 등 입력 x의 차원에 맞게 view를 조정합니다.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    # 각 샘플에 대해 랜덤하게 drop할지 결정합니다.
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 버림 연산으로 0 또는 1로 만듭니다.
    # 살아남은 path의 기댓값을 보존하기 위해 keep_prob으로 나눠줍니다.
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim: int, r_expansion: int = 4, kernel_size: int = 7, drop_path_rate: float = 0.):
        super().__init__()
        
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        
        self.pwconv1 = nn.Linear(dim, dim * r_expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * r_expansion, dim)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)
        
        x = self.drop_path(x)
        
        return x + identity

# Feature_Extractor의 역연산을 수행하는 ConvNeXt 스타일 Decoder (개선된 버전)
class Decoder(nn.Module):
    """
    ConvNeXt-style Decoder that is independent of initial_input_length.
    It uses nn.Upsample + nn.Conv1d instead of nn.ConvTranspose1d for robust length reconstruction.
    """
    def __init__(self, conv_configs: List[Tuple[int, int, int]], drop_path_rate: float = 0.0):
        super().__init__()
        
        self.decoder_blocks = nn.ModuleList()
        
        reversed_configs = conv_configs[::-1]

        in_ch = reversed_configs[0][0]

        for i, (enc_out_ch, kernel_size, stride) in enumerate(reversed_configs):
            is_last_layer = (i == len(reversed_configs) - 1)
            out_ch = 1 if is_last_layer else reversed_configs[i+1][0]

            block = nn.Sequential()

            if stride > 1:
                block.add_module("upsample", nn.Upsample(scale_factor=stride, mode='nearest'))

            if not is_last_layer:
                block.add_module("convnext", ConvNeXtBlock1D(
                    dim=out_ch,
                    kernel_size=7,
                    drop_path_rate=drop_path_rate
                ))
            
            self.decoder_blocks.append(block)
            
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor from the feature extractor. 
                              Shape: (Batch, Channels, Length)
        Returns:
            torch.Tensor: The reconstructed signal. Shape: (Batch, 1, Original_Length)
        """
        for block in self.decoder_blocks:
            x = block(x)
        return x

if __name__ == '__main__':
    encoder_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (512, 2, 2)
    ]

    latent_features = torch.randn(4, 512, 49) 

    print(f"Original latent feature shape: {latent_features.shape}")

    # Decoder 생성
    decoder = Decoder(conv_configs=encoder_configs, drop_path_rate=0.2)
    decoder.train()

    # 신호 복원
    reconstructed_signal = decoder(latent_features)

    print(f"Reconstructed signal shape: {reconstructed_signal.shape}")