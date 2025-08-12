import torch
from torch import nn, einsum
from torch.nn import functional as F
import copy

# 이 파일이 models 폴더 내에 있다고 가정하고 상대 경로로 임포트합니다.
# 만약 다른 위치에 있다면 경로를 수정해야 합니다.
from models import WaveEncode
from models import Predictor

# EMA 업데이트 함수
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

        # Context Encoder
        self.context_encoder = encoder
        
        # Target Encoder (Context Encoder와 동일한 구조, 가중치는 EMA로 업데이트)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        
        # Predictor
        self.predictor = predictor

        # EMA 업데이트를 위한 객체
        self.ema_updater = EMA(ema_decay)

        # Target Encoder는 학습되지 않도록 설정
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        """
        Target Encoder의 가중치를 EMA 방식으로 업데이트합니다.
        """
        update_moving_average(self.ema_updater, self.target_encoder, self.context_encoder)

    def _create_masks(self, seq_len: int, device: torch.device):
        """
        Context와 Target을 위한 마스크 인덱스를 생성합니다.
        
        Args:
            seq_len (int): 전체 시퀀스 길이
            device (torch.device): 텐서가 생성될 디바이스

        Returns:
            tuple[torch.Tensor, torch.Tensor]: context_indices, target_indices
        """
        num_masked_spans = int(seq_len * self.mask_ratio / self.mask_span_length)
        if num_masked_spans == 0:
             # 마스킹할 스팬이 없는 경우 빈 텐서 반환
            return torch.arange(seq_len, device=device), torch.tensor([], dtype=torch.long, device=device)

        # 마스킹 시작 위치를 랜덤하게 선택
        masked_span_start = torch.randperm(seq_len - self.mask_span_length + 1, device=device)[:num_masked_spans]
        
        # 마스킹될 인덱스 생성
        masked_indices = masked_span_start[:, None] + torch.arange(self.mask_span_length, device=device)[None, :]
        masked_indices = masked_indices.flatten().clamp(max=seq_len - 1)
        masked_indices = torch.unique(masked_indices) # 중복 제거

        # 전체 인덱스 생성
        full_indices = torch.arange(seq_len, device=device)
        
        # 마스크 생성 (True: 보이는 부분, False: 마스킹된 부분)
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[masked_indices] = False
        
        # context와 target 인덱스 분리
        context_indices = full_indices[mask]
        target_indices = full_indices[~mask]
        
        return context_indices, target_indices

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        W_JEPA의 forward pass
        """
        # 1. Feature Extractor를 통해 전체 오디오에서 feature 추출
        features, new_lengths = self.context_encoder.feature_extractor(x, lengths)
        
        B, N, D = features.shape
        device = features.device
        
        # 배치 내에서 가장 짧은 길이를 기준으로 마스크 생성
        min_len = min(new_lengths).item()
        
        # 2. Context / Target 마스크 생성
        context_indices, target_indices = self._create_masks(min_len, device)
        
        # 배치 전체에 동일한 마스크를 적용하기 위해 확장
        context_indices = context_indices.unsqueeze(0).expand(B, -1)
        target_indices = target_indices.unsqueeze(0).expand(B, -1)
        
        # 3. 인덱스를 사용하여 context와 target feature 선택
        # gather를 사용하여 각 배치 샘플에서 해당 인덱스의 feature를 가져옴
        context_features = torch.gather(features, 1, context_indices.unsqueeze(-1).expand(-1, -1, D))
        
        # 4. Context Encoder는 context feature만 처리
        encoded_context = self.context_encoder.encode(context_features, mask=None)

        # 5. Target Encoder는 전체 feature를 처리하고 target 부분만 선택 (no_grad)
        with torch.no_grad():
            self._update_target_encoder() # EMA 업데이트
            # Target Encoder는 전체 feature를 한번에 처리
            full_encoded_features = self.target_encoder.encode(features, mask=None)
            encoded_target = torch.gather(full_encoded_features, 1, target_indices.unsqueeze(-1).expand(-1, -1, D))

        # 6. Predictor가 context를 기반으로 target 예측
        predicted_target = self.predictor.enc(encoded_context, context_indices, target_indices)
        
        # 7. Loss 계산 (Mean Squared Error)
        loss = F.l1_loss(predicted_target, encoded_target)
        # loss = F.mse_loss(predicted_target, encoded_target)
        # print(predicted_target.shape, encoded_target.shape)
        return loss, predicted_target, encoded_target

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
        
        predictor_dim = 512
        predictor_layers = 4
        predictor_heads = 8
        
        max_seq_len = 256

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
            mask_ratio=0.6,
            mask_span_length=10
        )
        
        # --- 더미 데이터 생성 ---
        batch_size = 4
        raw_audio_len = 16000 * 5
        
        x_input = torch.randn(batch_size, raw_audio_len)
        input_lengths = torch.tensor([raw_audio_len, raw_audio_len - 2000, raw_audio_len - 5000, raw_audio_len - 1000])

        # --- 모델 Forward Pass 및 학습 ---
        optimizer = torch.optim.AdamW(w_jepa_model.parameters(), lr=3e-3)

        print("--- Start Training Loop (Example) ---")
        for i in range(5):
            optimizer.zero_grad()
            
            loss, _, _ = w_jepa_model(x_input, input_lengths)
            
            loss.backward()
            optimizer.step()
            
            print(f"Step {i+1}, Loss: {loss.item()}")
        print("--- End Training Loop ---")

    main()