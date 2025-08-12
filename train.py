import torch
from torch import nn, einsum
from torch.nn import functional as F
import copy
import torchaudio
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math

# --- 헬퍼 함수 및 클래스 (사용자 제공) ---

@torch.no_grad()
def update_moving_average(ema_updater, ma_model, current_model):
    """
    Exponential Moving Average(EMA)를 사용하여 모델의 가중치를 업데이트합니다.
    ma_model의 가중치가 current_model의 가중치를 향해 점진적으로 업데이트됩니다.
    """
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class EMA():
    """
    EMA 계산을 위한 헬퍼 클래스
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# --- 모델 정의 (사용자 제공 코드 기반) ---

# 참고: 'models' 모듈을 사용할 수 없으므로, 스크립트 실행을 위해
# WaveEncode와 Predictor의 모의(mock) 클래스를 정의합니다.
# 실제 학습 시에는 이 부분을 삭제하고 원래의 import 구문을 사용하세요.
class WaveEncode(nn.Module):
    """
    모의 WaveEncode 클래스.
    실제 모델로 교체해야 합니다.
    """
    def __init__(self, conv_configs, nlayers, dim, nhead, ratio):
        super().__init__()
        # 실제 WaveNet/CNN 기반 Feature Extractor가 와야 할 자리
        self.feature_extractor_mock = nn.Conv1d(1, dim, kernel_size=160, stride=160)
        # 실제 Transformer Encoder가 와야 할 자리
        self.encoder_mock = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=int(dim * ratio), batch_first=True),
            num_layers=nlayers
        )
        self.dim = dim

    def feature_extractor(self, x, lengths):
        # (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.feature_extractor_mock(x)
        # (B, C, N) -> (B, N, C)
        features = features.transpose(1, 2)
        
        # stride에 따라 길이 다시 계산
        new_lengths = torch.div(lengths, 160, rounding_mode='floor')
        return features, new_lengths

    def encode(self, x, mask=None):
        return self.encoder_mock(x, src_key_padding_mask=mask)


class Predictor(nn.Module):
    """
    모의 Predictor 클래스.
    실제 모델로 교체해야 합니다.
    """
    def __init__(self, in_dim, dim, nlayers, nhead, ratio, max_seq_len):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, in_dim)
        self.predictor_mock = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=int(dim*ratio), batch_first=True),
            num_layers=nlayers
        )
        # 최종 출력을 target의 dimension과 맞추기 위한 레이어
        self.out_proj = nn.Linear(dim, in_dim)

    def enc(self, context_features, context_indices, target_indices):
        # 위치 임베딩 추가
        context_pos = self.pos_embedding(context_indices)
        target_pos = self.pos_embedding(target_indices)
        
        x = context_features + context_pos
        
        # 예측 (Transformer Decoder와 유사한 역할)
        predicted_sequence = self.predictor_mock(x)
        
        # target 위치에 해당하는 결과만 선택 (단순화된 예시)
        mean_context = predicted_sequence.mean(dim=1, keepdim=True)
        predicted_target = mean_context.expand(-1, target_indices.shape[1], -1)
        predicted_target = predicted_target + target_pos
        
        return self.out_proj(predicted_target)


class W_JEPA(nn.Module):
    """
    Wave-JEPA 모델 클래스. Context Encoder, Target Encoder, Predictor를 포함하며
    I-JEPA 논문의 아키텍처를 따릅니다.
    """
    def __init__(
        self,
        encoder: WaveEncode,
        predictor: Predictor,
        ema_decay: float = 0.996,
        mask_ratio: float = 0.6,
        mask_span_length: int = 10,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_span_length = mask_span_length
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
        num_masked_spans = int(seq_len * self.mask_ratio / self.mask_span_length)
        if num_masked_spans == 0:
            return torch.arange(seq_len, device=device), torch.tensor([], dtype=torch.long, device=device)

        masked_span_start = torch.randperm(seq_len - self.mask_span_length + 1, device=device)[:num_masked_spans]
        masked_indices = masked_span_start[:, None] + torch.arange(self.mask_span_length, device=device)[None, :]
        masked_indices = masked_indices.flatten().clamp(max=seq_len - 1)
        masked_indices = torch.unique(masked_indices)

        full_indices = torch.arange(seq_len, device=device)
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[masked_indices] = False
        
        context_indices = full_indices[mask]
        target_indices = full_indices[~mask]
        
        return context_indices, target_indices

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        features, new_lengths = self.context_encoder.feature_extractor(x, lengths)
        B, N, D = features.shape
        device = features.device
        
        min_len = min(new_lengths).item()
        
        context_indices, target_indices = self._create_masks(min_len, device)
        
        context_indices = context_indices.unsqueeze(0).expand(B, -1)
        target_indices = target_indices.unsqueeze(0).expand(B, -1)
        
        context_features = torch.gather(features, 1, context_indices.unsqueeze(-1).expand(-1, -1, D))
        
        encoded_context = self.context_encoder.encode(context_features, mask=None)

        with torch.no_grad():
            self._update_target_encoder()
            full_encoded_features = self.target_encoder.encode(features, mask=None)
            encoded_target = torch.gather(full_encoded_features, 1, target_indices.unsqueeze(-1).expand(-1, -1, D))

        predicted_target = self.predictor.enc(encoded_context, context_indices, target_indices)
        
        loss = F.l1_loss(predicted_target, encoded_target)
        
        return loss, predicted_target, encoded_target

# --- 데이터 처리 및 학습 루프 ---

def collate_fn(batch, sample_rate=16000, duration=10):
    """
    DataLoader를 위한 커스텀 collate 함수.
    오디오를 10초로 패딩 또는 랜덤 트렁케이트합니다.
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
            # Warm-up: 선형적으로 학습률 증가
            return float(current_iter) / float(max(1, warmup_iter))
        else:
            # Cosine Decay: 학습률 감소
            progress = float(current_iter - warmup_iter) / float(max(1, max_iter - warmup_iter))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 최종 학습률에 도달하도록 스케일링
            scaled_lr = final_lr + (base_lr - final_lr) * cosine_decay
            return scaled_lr / base_lr # LambdaLR은 base_lr에 곱해지는 비율을 반환해야 함

    return LambdaLR(optimizer, lr_lambda)

def main():
    # --- 학습 하이퍼파라미터 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 8
    base_learning_rate = 4e-3
    final_learning_rate = 1e-8
    num_epochs = 10 # 10 에포크로 변경
    warmup_iter = 500 # Warm-up 이터레이션
    
    # --- 모델 하이퍼파라미터 설정 ---
    conv_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (512, 2, 2)
    ]
    encoder_dim = 512
    encoder_layers = 6
    encoder_heads = 8
    
    predictor_dim = 512
    predictor_layers = 4
    predictor_heads = 8
    
    max_seq_len = 10_000

    # --- 데이터셋 및 DataLoader 준비 ---
    print("Loading LibriSpeech dataset...")
    try:
        train_dataset = torchaudio.datasets.LIBRISPEECH(
            root='./data',
            url='train-clean-100',
            download=True
        )
    except Exception as e:
        print(f"Failed to download or load dataset: {e}")
        print("Please check your internet connection or data directory permissions.")
        return

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # --- 모델 초기화 ---
    encoder = WaveEncode(
        conv_configs=conv_configs, 
        nlayers=encoder_layers, 
        dim=encoder_dim, 
        nhead=encoder_heads, 
        ratio=4.0
    )

    predictor = Predictor(
        in_dim=encoder_dim, 
        dim=predictor_dim, 
        nlayers=predictor_layers, 
        nhead=predictor_heads, 
        ratio=4.0, 
        max_seq_len=max_seq_len
    )

    w_jepa_model = W_JEPA(
        encoder=encoder,
        predictor=predictor,
        mask_ratio=0.75,
        mask_span_length=10
    ).to(device)
    
    optimizer = torch.optim.AdamW(w_jepa_model.parameters(), lr=base_learning_rate)
    
    # --- 스케줄러 설정 ---
    max_iter = len(train_loader) * num_epochs
    scheduler = get_lr_scheduler(optimizer, max_iter, warmup_iter, final_learning_rate)
    
    # --- 학습 루프 ---
    print("--- Start Training Loop ---")
    current_iter = 0
    for epoch in range(num_epochs):
        w_jepa_model.train()
        total_loss = 0
        
        # tqdm을 이용한 progress bar 설정
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for waveforms, lengths in pbar:
            waveforms = waveforms.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            loss, _, _ = w_jepa_model(waveforms, lengths)
            
            loss.backward()
            optimizer.step()
            scheduler.step() # 이터레이션마다 스케줄러 업데이트
            
            total_loss += loss.item()
            current_iter += 1
            
            # tqdm progress bar에 현재 손실과 학습률 표시
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Finished, Average Loss: {avg_loss:.4f} ---")

    print("--- End Training Loop ---")


if __name__ == '__main__':
    main()