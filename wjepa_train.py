import torch
from torch import nn, einsum
from torch.nn import functional as F
import copy
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math
import random
import warnings
import yaml


from torch.amp import autocast, GradScaler
from models import WaveEncode, Predictor
from utils import MaskCollator

# --- 특정 UserWarning 무시 ---
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec")
# warnings.filterwarnings("ignore", message="torch.nn.functional. अभी तक कार्यान्वित नहीं है for 'gumbel_softmax'") << ?? Gemini 2.5 pro Hallusination

# --- AverageMeter 클래스 정의 ---
class AverageMeter:
    """Computes and stores the average, current value, and history of values."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        self.history.append(val)


# --- 헬퍼 함수 및 클래스 ---

@torch.no_grad()
def update_moving_average(ema_updater, ma_model, current_model):
    """
    Exponential Moving Average(EMA)를 사용하여 모델의 가중치를 업데이트합니다.
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

# --- 모델 정의 ---


class W_JEPA(nn.Module):
    """
    Wave-JEPA 모델 클래스. I-JEPA 논문의 아키텍처를 따릅니다.
    wjepa_mask.py의 MaskCollator를 사용하여 마스킹을 수행합니다.
    """
    def __init__(
        self,
        encoder: WaveEncode,
        predictor: Predictor,
        ema_decay: float = 0.996,
        # --- MaskCollator 파라미터 ---
        seq_length: int = 160000,  # 10초 오디오 (16000Hz * 10s)
        patch_size: int = 320,     # Conv 레이어의 총 스트라이드
        enc_mask_scale: tuple = (0.85, 1.0),
        pred_mask_scale: tuple = (0.15, 0.2),
        npred: int = 4,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = predictor
        self.ema_updater = EMA(ema_decay)
        
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # wjepa_mask.py의 MaskCollator 초기화
        self.mask_collator = MaskCollator(
            seq_length=seq_length,
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            nenc=1,
            npred=npred,
            allow_overlap=False,
            mask_strategy='contiguous_blocks',
            # 스팬을 크게 설정하여 단일 블록처럼 작동하도록 유도
            pred_span_scale=(10, 20),
            enc_span_scale=(100, 200),
        )

    @torch.no_grad()
    def _update_target_encoder(self):
        update_moving_average(self.ema_updater, self.target_encoder, self.context_encoder)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        feature, new_lengths = self.context_encoder.feature_extractor(x, lengths)
        B, N, D_in = feature.shape
        device = x.device

        # MaskCollator를 사용하여 인코더(컨텍스트) 및 예측기(타겟) 마스크 생성
        # collator는 원본 오디오 길이를 기대하므로 `lengths`를 전달합니다.
        _, enc_masks, pred_masks = self.mask_collator(feature, lengths=lengths)
        enc_masks = enc_masks.to(device)
        pred_masks = pred_masks.to(device)

        # 컨텍스트 마스크 (B, 1, M_enc) -> (B, M_enc)
        context_indices = enc_masks.squeeze(1)

        if context_indices.numel() == 0:
            return torch.tensor(0., device=device, requires_grad=True), None, None

        # 컨텍스트 인코딩
        gather_indices_context = context_indices.unsqueeze(-1).expand(-1, -1, D_in)
        context_features = torch.gather(feature, 1, gather_indices_context)
        encoded_context = self.context_encoder.encode(context_features, lengths=None)

        # 타겟 인코더로 전체 시퀀스에 대한 임베딩 계산
        with torch.no_grad():
            if self.training:
                self._update_target_encoder()
            full_encoded_features = self.target_encoder.encode(feature, lengths=new_lengths.to(device))

        total_loss = 0.0
        valid_masks = 0
        
        # 여러 타겟 마스크에 대해 손실 계산
        num_target_masks = pred_masks.shape[1]
        for i in range(num_target_masks):
            target_indices = pred_masks[:, i, :]  # (B, M_pred)

            if target_indices.numel() == 0:
                continue
            
            valid_masks += 1

            # 타겟 임베딩 추출
            with torch.no_grad():
                gather_indices_target = target_indices.unsqueeze(-1).expand(-1, -1, D_in)
                encoded_target = torch.gather(full_encoded_features, 1, gather_indices_target)

            # 타겟 예측
            predicted_target = self.predictor.enc(encoded_context, context_indices, target_indices)
            
            # 손실 계산
            loss = F.mse_loss(predicted_target, encoded_target)
            total_loss += loss

        # 타겟 블록들의 평균 손실
        if valid_masks > 0:
            avg_loss = total_loss / valid_masks
        else:
            # 타겟이 하나도 없는 경우
            return torch.tensor(0., device=device, requires_grad=True), None, None
        
        # 원래 forward와 출력 형식 맞추기 (predicted/encoded target은 대표값 없으므로 None)
        return avg_loss, None, None

def get_lr_scheduler(optimizer, max_iter, warmup_iter, final_lr):
    """
    Creates a learning rate scheduler with a linear warmup followed by a cosine decay.
    After max_iter, the learning rate is held constant at final_lr.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        max_iter (int): The total number of iterations for training decay.
        warmup_iter (int): The number of iterations for the linear warmup phase.
        final_lr (float): The final learning rate after decay.
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """
    # The base learning rate is fetched from the optimizer's parameter groups.
    base_lr = optimizer.param_groups[0]['lr']
    
    # This is the lambda function that will calculate the learning rate multiplier for each step.
    def lr_lambda(current_iter):
        # During the warmup phase, the learning rate increases linearly.
        if current_iter < warmup_iter:
            return float(current_iter) / float(max(1, warmup_iter))
        # If the current iteration has reached or passed max_iter, hold the final learning rate.
        elif current_iter >= max_iter:
            return final_lr / base_lr
        # After the warmup and before max_iter, the learning rate follows a cosine decay schedule.
        else:
            # Calculate the progress of the decay phase (from 0.0 to 1.0).
            progress = float(current_iter - warmup_iter) / float(max(1, max_iter - warmup_iter))
            # Apply the cosine decay formula.
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale the learning rate between the base_lr and final_lr.
            scaled_lr = final_lr + (base_lr - final_lr) * cosine_decay
            # Return the multiplier relative to the base_lr.
            return scaled_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)
