import torch
from torch import nn, einsum
from torch.nn import functional as F
import copy
import torchaudio
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math
import warnings
import random

from models import WaveEncode, Block, Predictor

# --- 특정 UserWarning 무시 ---
# torchaudio 백엔드 변경 관련 경고 메시지를 무시합니다.
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec")


# --- 헬퍼 함수 및 클래스 ---

@torch.no_grad()
def update_moving_average(ema_updater, ma_model, current_model):
    """
    EMA(Exponential Moving Average)를 사용하여 이동 평균 모델의 가중치를 업데이트합니다.
    """
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class EMA():
    """
    EMA 계산을 위한 헬퍼 클래스입니다.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class W_JEPA(nn.Module):
    """
    I-JEPA 아키텍처를 따르는 Wave-JEPA 모델 클래스입니다.
    """
    def __init__(
        self,
        encoder: WaveEncode,
        predictor: Predictor,
        ema_decay: float = 0.996,
        # --- I-JEPA 논문에 따른 마스킹 파라미터로 변경 ---
        num_target_blocks: int = 4,
        target_block_scale: tuple[float, float] = (0.15, 0.2),
        context_block_scale: tuple[float, float] = (0.85, 1.0),
    ):
        super().__init__()
        # --- 새로운 마스킹 파라미터 저장 ---
        self.num_target_blocks = num_target_blocks
        self.target_block_scale = target_block_scale
        self.context_block_scale = context_block_scale
        
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = predictor
        self.ema_updater = EMA(ema_decay)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        update_moving_average(self.ema_updater, self.target_encoder, self.context_encoder)

    def _create_masks(self, seq_len: int, device: torch.device):
        """
        I-JEPA 논문의 multi-block 마스킹 전략을 구현합니다.
        1. 여러 개의 분리된 '타겟' 블록을 샘플링합니다.
        2. 하나의 넓은 '컨텍스트' 블록을 샘플링합니다.
        3. 컨텍스트에서 타겟과 겹치는 부분을 제거합니다.
        """
        full_indices = torch.arange(seq_len, device=device)
        
        # --- 1. 타겟 블록 샘플링 ---
        target_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for _ in range(self.num_target_blocks):
            # 타겟 블록의 길이(scale)를 랜덤하게 결정
            block_len = int(seq_len * (self.target_block_scale[0] + random.random() * (self.target_block_scale[1] - self.target_block_scale[0])))
            if block_len == 0: continue
            
            # 타겟 블록의 시작 위치를 랜덤하게 결정
            start_idx = torch.randint(0, seq_len - block_len + 1, (1,), device=device).item()
            target_mask[start_idx : start_idx + block_len] = True

        # --- 2. 컨텍스트 블록 샘플링 ---
        context_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        # 컨텍스트 블록의 길이를 랜덤하게 결정
        context_len = int(seq_len * (self.context_block_scale[0] + random.random() * (self.context_block_scale[1] - self.context_block_scale[0])))
        if context_len > 0:
            # 컨텍스트 블록의 시작 위치를 랜덤하게 결정
            start_idx = torch.randint(0, seq_len - context_len + 1, (1,), device=device).item()
            context_mask[start_idx : start_idx + context_len] = True

        # --- 3. 컨텍스트에서 타겟과 겹치는 부분 제거 ---
        context_mask = context_mask & (~target_mask)

        # 마스크(boolean)를 인덱스(long)로 변환
        context_indices = full_indices[context_mask]
        target_indices = full_indices[target_mask]

        return context_indices, target_indices


    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        features, new_lengths = self.context_encoder.feature_extractor(x, lengths)
        B, N, D = features.shape
        device = features.device
        
        # 배치 내에서 가장 짧은 길이를 기준으로 마스크 생성
        min_len = min(new_lengths).item()
        
        context_indices, target_indices = self._create_masks(min_len, device)
        
        # 예측할 타겟이 없으면 loss 0 반환
        if target_indices.numel() == 0 or context_indices.numel() == 0:
            return torch.tensor(0., device=device, requires_grad=True), None, None

        # 생성된 마스크를 배치 전체에 동일하게 적용
        context_indices = context_indices.unsqueeze(0).expand(B, -1)
        target_indices = target_indices.unsqueeze(0).expand(B, -1)
        
        # 컨텍스트 인덱스를 사용해 컨텍스트 특징 추출
        context_features = torch.gather(features, 1, context_indices.unsqueeze(-1).expand(-1, -1, D))
        
        # 컨텍스트 인코딩
        encoded_context = self.context_encoder.encode(context_features, mask=None)

        with torch.no_grad():
            self._update_target_encoder()
            # 타겟 인코더는 전체 특징을 인코딩
            full_encoded_features = self.target_encoder.encode(features, mask=None)
            # 타겟 인덱스를 사용해 정답(타겟) 특징 추출
            encoded_target = torch.gather(full_encoded_features, 1, target_indices.unsqueeze(-1).expand(-1, -1, D))

        # Predictor를 사용해 타겟 특징 예측
        predicted_target = self.predictor.enc(encoded_context, context_indices, target_indices)
        
        # MSE Loss 계산
        loss = F.mse_loss(predicted_target, encoded_target)
        
        return loss, predicted_target, encoded_target

# --- 데이터 처리 및 학습 루프 ---

def collate_fn(batch, sample_rate=16000, duration=10):
    """
    DataLoader를 위한 커스텀 collate 함수.
    오디오를 10초로 패딩하거나 랜덤하게 자릅니다.
    """
    target_len = sample_rate * duration
    
    waveforms = []
    lengths = []
    
    for (waveform, sr, _, _, _, _) in batch:
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        current_len = waveform.shape[1]
        
        if current_len < target_len:
            padding = target_len - current_len
            padded_waveform = F.pad(waveform, (0, padding))
            waveforms.append(padded_waveform)
            lengths.append(current_len)
        else:
            start = torch.randint(0, current_len - target_len + 1, (1,)).item()
            truncated_waveform = waveform[:, start:start + target_len]
            waveforms.append(truncated_waveform)
            lengths.append(target_len)
            
    waveforms_tensor = torch.cat(waveforms, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return waveforms_tensor, lengths_tensor


def get_lr_scheduler(optimizer, max_iter, warmup_iter, final_lr):
    """ Warm-up 후 Cosine Decay를 적용하는 LambdaLR 스케줄러를 생성합니다. """
    base_lr = optimizer.param_groups[0]['lr']
    
    def lr_lambda(current_iter):
        if current_iter < warmup_iter:
            # Warm-up: 선형 증가
            return float(current_iter) / float(max(1, warmup_iter))
        else:
            # Cosine Decay: 학습률 감소
            progress = float(current_iter - warmup_iter) / float(max(1, max_iter - warmup_iter))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 최종 학습률에 맞게 스케일링
            scaled_lr = final_lr + (base_lr - final_lr) * cosine_decay
            return scaled_lr / base_lr # LambdaLR는 base_lr에 대한 승수를 필요로 함

    return LambdaLR(optimizer, lr_lambda)

# --- 모델 실행 및 학습 예제 ---
if __name__ == '__main__':
    def main():
        # --- 모델 하이퍼파라미터 설정 ---
        conv_configs = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
            (512, 3, 2), (512, 2, 2), (512, 2, 2)
        ]
        
        encoder_dim = 512
        encoder_layers = 6
        encoder_heads = 8
        
        # --- Predictor의 차원을 Encoder보다 작게 설정 (병목 구조) ---
        predictor_dim = 512 
        predictor_layers = 4
        predictor_heads = 8
        
        # Feature Extractor를 통과한 후의 최대 시퀀스 길이를 예측하여 설정
        # 이 값은 실제 데이터와 conv_configs에 따라 달라지므로, 넉넉하게 설정하는 것이 좋습니다.
        max_seq_len = 500 

        # --- 모델 초기화 ---
        encoder = WaveEncode(
            conv_configs=conv_configs, 
            nlayers=encoder_layers, 
            dim=encoder_dim, 
            nhead=encoder_heads, 
            ratio=4.0,
        )

        predictor = Predictor(
            in_dim=encoder_dim, 
            dim=predictor_dim, 
            nlayers=predictor_layers, 
            nhead=predictor_heads, 
            ratio=4.0, 
            max_seq_len=max_seq_len,
        )

        # --- W_JEPA 모델 생성 (I-JEPA 기본 마스킹 파라미터 사용) ---
        w_jepa_model = W_JEPA(
            encoder=encoder,
            predictor=predictor,
            # num_target_blocks, target_block_scale 등은 기본값 사용
        )
        
        # --- 더미 데이터 생성 ---
        batch_size = 4 # 메모리 사용량을 고려하여 배치 사이즈 조정
        raw_audio_len = 16000 * 10 # 10초 길이의 오디오
        
        x_input = torch.randn(batch_size, raw_audio_len)
        input_lengths = torch.full((batch_size,), raw_audio_len, dtype=torch.long)
        input_lengths[1] -= 2000
        input_lengths[2] -= 5000
        
        # --- 모델 Forward Pass 및 학습 ---
        optimizer = torch.optim.AdamW(w_jepa_model.parameters(), lr=1e-5)

        print("--- Start Training Loop (Example) ---")
        for i in range(10): # 반복 횟수 증가
            optimizer.zero_grad()
            
            loss, _, _ = w_jepa_model(x_input, input_lengths)
            
            loss.backward()
            optimizer.step()
            
            print(f"Step {i+1}, Loss: {loss.item()}")
        print("--- End Training Loop ---")

    main()
