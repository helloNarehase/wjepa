import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ConvNeXtBlock1D(nn.Module):
    """
    ConvNeXt Block for 1D data.
    - r_expansion: Inverted bottleneck expansion ratio.
    - kernel_size: Kernel size for the depthwise convolution.
    - drop_p: Dropout probability.
    """
    def __init__(self, dim: int, r_expansion: int = 4, kernel_size: int = 7, drop_p: float = 0.):
        super().__init__()
        
        # 1. Depthwise Convolution: 각 채널을 독립적으로 처리하여 공간/시간적 특징을 학습합니다.
        # 'padding="same"'을 사용하여 입력과 출력의 길이를 동일하게 유지합니다.
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding="same", groups=dim
        )
        
        # 2. Layer Normalization: 배치 크기에 의존하지 않는 정규화 방식입니다.
        # ConvNeXt는 BatchNorm 대신 LayerNorm을 사용합니다.
        self.norm = nn.LayerNorm(dim)
        
        # 3. Pointwise Convolutions (Inverted Bottleneck):
        # 채널 수를 늘렸다가(expansion) 다시 줄여서(projection) 표현력을 높입니다.
        # nn.Linear를 사용하여 1x1 Conv와 동일한 연산을 수행합니다.
        self.pwconv1 = nn.Linear(dim, dim * r_expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * r_expansion, dim)
        
        # 4. Dropout (Stochastic Depth와 유사한 역할)
        self.drop = nn.Dropout(drop_p) if drop_p > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 잔차 연결(Residual Connection)을 위해 원본 입력을 저장합니다.
        identity = x
        
        x = self.dwconv(x)
        
        # LayerNorm과 Linear 연산을 위해 텐서의 차원을 변경합니다 (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # 다시 원래 차원으로 복원합니다 (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)
        
        x = self.drop(x)
        
        # 잔차 연결을 통해 입력 정보를 더해줍니다.
        return x + identity

# Feature_Extractor의 역연산을 수행하는 ConvNeXt 스타일 Decoder
class Decoder(nn.Module):
    def __init__(self, conv_configs: List[Tuple[int, int, int]], dropout: float = 0.0, initial_input_length: int = 16000):
        super().__init__()
        
        self.decoder_layers = nn.ModuleList()
        
        # --- Encoder의 각 레이어 입력 길이를 미리 계산 ---
        encoder_input_lengths = []
        current_length = initial_input_length
        for _, kernel_size, stride in conv_configs:
            encoder_input_lengths.append(current_length)
            # Feature_Extractor의 길이 계산 공식
            current_length = (current_length - kernel_size) // stride + 1
        
        # Decoder는 Encoder의 역순으로 구성되므로, 설정과 길이 리스트를 뒤집습니다.
        reversed_configs = conv_configs[::-1]
        reversed_encoder_input_lengths = encoder_input_lengths[::-1]

        # 첫 번째 레이어의 입력 채널은 Encoder의 마지막 출력 채널과 같습니다.
        in_ch = reversed_configs[0][0]

        for i, (enc_out_ch, kernel_size, stride) in enumerate(reversed_configs):
            # Decoder의 출력 채널은 대칭되는 Encoder의 입력 채널과 같습니다.
            if i == len(reversed_configs) - 1:
                out_ch = 1
            else:
                out_ch = reversed_configs[i+1][0]

            # --- ConvTranspose1d 파라미터를 정밀하게 계산 ---
            # 목표 출력 길이는 대칭되는 Encoder 레이어의 '입력' 길이입니다.
            target_len = reversed_encoder_input_lengths[i]
            
            # output_padding = (L_in - K) % S 공식을 사용하여 손실된 길이를 보정합니다.
            output_padding = (target_len - kernel_size) % stride

            # 1. Upsampling Layer: ConvTranspose1d를 사용하여 길이를 복원합니다.
            # Encoder와 동일한 kernel_size, stride를 사용하고 계산된 output_padding을 적용합니다.
            upsample_layer = nn.ConvTranspose1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=0, # Padding은 0으로 고정하여 계산을 단순화합니다.
                output_padding=output_padding
            )
            
            # 2. ConvNeXt Block: 업샘플링된 특징을 처리합니다.
            convnext_block = ConvNeXtBlock1D(
                dim=out_ch,
                kernel_size=7, # ConvNeXt에서 주로 사용하는 큰 커널 크기
                drop_p=dropout
            )
            
            self.decoder_layers.append(upsample_layer)
            self.decoder_layers.append(convnext_block)
            
            # 다음 레이어의 입력 채널을 현재 레이어의 출력 채널로 업데이트합니다.
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor from the feature extractor. 
                              Shape: (Batch, Channels, Length)
        Returns:
            torch.Tensor: The reconstructed signal. Shape: (Batch, 1, Original_Length)
        """
        for layer in self.decoder_layers:
            x = layer(x)
        return x