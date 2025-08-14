import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from models import Feature_Extractor, Decoder
# --- 검증을 위한 실행 코드 ---
if __name__ == '__main__':
    # --- 설정 및 데이터 ---
    conv_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (512, 2, 2)
    ]
    
    initial_length = 12000
    dummy_input = torch.randn(4, 1, initial_length)
    input_lengths = torch.tensor([initial_length] * 4)
    
    # --- Encoder 실행 ---
    encoder = Feature_Extractor(conv_configs)
    encoded_features, encoded_lengths = encoder(dummy_input, input_lengths)
    
    print(f"Encoder Input Shape: {dummy_input.shape}")
    print(f"Encoded Feature Shape: {encoded_features.shape}")
    print(f"Encoded Lengths: {encoded_lengths}")
    print("-" * 30)

    # --- Decoder 실행 ---
    # Decoder 생성 시 원본 길이를 알려줍니다.
    decoder = Decoder(conv_configs, initial_input_length=initial_length)
    reconstructed_signal = decoder(encoded_features)
    
    print(f"Decoder Input Shape (Encoded Features): {encoded_features.shape}")
    print(f"Reconstructed Signal Shape: {reconstructed_signal.shape}")

    # 모델의 파라미터 수 확인
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Number of parameters in Decoder: {num_params:,}")
