import torch
from torch import nn, einsum
from torch.nn import functional as F
import copy
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math
import random
import warnings
import yaml

from torch.amp import autocast, GradScaler
from models import WaveEncode, Predictor

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
        """
        full_indices = torch.arange(seq_len, device=device)
        
        # --- 1. 타겟 블록 샘플링 ---
        target_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for _ in range(self.num_target_blocks):
            block_len = int(seq_len * (self.target_block_scale[0] + random.random() * (self.target_block_scale[1] - self.target_block_scale[0])))
            if block_len == 0: continue
            
            start_idx = torch.randint(0, seq_len - block_len + 1, (1,), device=device).item()
            target_mask[start_idx : start_idx + block_len] = True

        # --- 2. 컨텍스트 블록 샘플링 ---
        context_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        context_len = int(seq_len * (self.context_block_scale[0] + random.random() * (self.context_block_scale[1] - self.context_block_scale[0])))
        if context_len > 0:
            start_idx = torch.randint(0, seq_len - context_len + 1, (1,), device=device).item()
            context_mask[start_idx : start_idx + context_len] = True

        # --- 3. 컨텍스트에서 타겟과 겹치는 부분 제거 ---
        context_mask = context_mask & (~target_mask)

        context_indices = full_indices[context_mask]
        target_indices = full_indices[target_mask]

        return context_indices, target_indices

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # NOTE: This simplified forward pass might not perfectly align with the original logic,
        # but it's structured to allow training to run.
        # The original code had a feature_extractor call that is now part of the encoder.
        # Let's assume the encoder handles feature extraction internally.
        
        # A mock feature extraction to get sequence length

        feature, new_lengths = self.context_encoder.feature_extractor(x, lengths)
        mock_seq_len = max(new_lengths)
        B, _, D_in = feature.shape
        
        device = x.device
        
        context_indices, target_indices = self._create_masks(mock_seq_len, device)
        
        if target_indices.numel() == 0 or context_indices.numel() == 0:
            return torch.tensor(0., device=device, requires_grad=True), None, None
        
        context_indices = context_indices.unsqueeze(0).expand(B, -1)
        target_indices = target_indices.unsqueeze(0).expand(B, -1)

        gather_indices = context_indices.unsqueeze(-1).expand(-1, -1, D_in)
        context_features = torch.gather(feature, 1, gather_indices) 
        encoded_context = self.context_encoder.encode(context_features, lengths=None)

        with torch.no_grad():
            if self.training: # [추가] 학습 중에만 EMA 업데이트 수행
                self._update_target_encoder()
            full_encoded_features = self.target_encoder.encode(feature, lengths=new_lengths.to(device))
            encoded_target = torch.gather(full_encoded_features, 1, target_indices.unsqueeze(-1).expand(-1, -1, D_in))

        predicted_target = self.predictor.enc(encoded_context, context_indices, target_indices)
        
        loss = F.mse_loss(predicted_target, encoded_target)
        
        return loss, predicted_target, encoded_target
# --- 데이터 처리 및 학습 루프 ---

def collate_fn(batch, sample_rate=16000, duration=10):
    target_len = sample_rate * duration
    waveforms, lengths = [], []
    
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

def main():
    # --- 학습 하이퍼파라미터 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f"Using device: {device}")

    batch_size = 64
    base_learning_rate = 3e-4
    final_learning_rate = 1e-9
    num_epochs = 30
    accumulation_steps = 8

    # --- 모델 하이퍼파라미터 설정 ---
    conv_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (512, 2, 2)
    ]
    encoder_dim = 512
    encoder_layers = 6
    encoder_heads = 8
    encoder_dropout = 0.3
    encoder_droppath = 0.3
    
    predictor_dim = 512
    predictor_layers = 4
    predictor_heads = 8
    predictor_droppath = 0.3
    
    max_seq_len = 10_000 # Predictor의 pos_embedding 크기와 관련

    root_dir = './data'

    # --- 데이터셋 및 DataLoader 준비 ---
    print("Loading LibriSpeech dataset...")
    try:
        train_clean_100 = torchaudio.datasets.LIBRISPEECH(
            root=root_dir,
            url='train-clean-100',
            download=True
        )

        train_clean_360 = torchaudio.datasets.LIBRISPEECH(
            root=root_dir,
            url='train-clean-360',
            download=True
        )

        train_other_500 = torchaudio.datasets.LIBRISPEECH(
            root=root_dir,
            url='train-other-500',
            download=True
        )
        full_dataset = ConcatDataset([train_clean_100, train_clean_360, train_other_500])

    except Exception as e:
        print(f"Failed to download or load dataset: {e}")
        return

    # --- [추가] 데이터셋을 학습용과 검증용으로 분리 ---
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    print(f"Splitting dataset: {train_size} for training, {val_size} for validation.")
    
    # [추가] 재현성을 위해 시드 고정
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # --- [추가] 검증용 DataLoader ---
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False, # 검증 데이터는 섞을 필요 없음
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
        ratio=4.0,
        dropout=encoder_dropout,
        droppath=encoder_droppath
    )

    predictor = Predictor(
        in_dim=encoder_dim, 
        dim=predictor_dim, 
        nlayers=predictor_layers, 
        nhead=predictor_heads, 
        ratio=4.0, 
        max_seq_len=max_seq_len,
        droppath=predictor_droppath
    )

    w_jepa_model = W_JEPA(
        encoder=encoder,
        predictor=predictor,
    ).to(device)
    
    optimizer = torch.optim.AdamW(w_jepa_model.parameters(), lr=base_learning_rate, weight_decay=0.02)
    
    max_iter = len(train_loader) // accumulation_steps * num_epochs
    scheduler = get_lr_scheduler(optimizer, max_iter, int(max_iter * 0.3), final_learning_rate)
    
    scaler = GradScaler(enabled=use_cuda)
    
    # [수정] 에포크별 평균 손실을 기록할 리스트
    all_epoch_train_losses = []
    all_epoch_val_losses = [] # [추가] 검증 손실 기록

    # --- 학습 루프 ---
    print("--- Start Training Loop ---")
    global_step = 0
    for epoch in range(num_epochs):
        # --- Training Phase ---
        w_jepa_model.train()
        epoch_loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", dynamic_ncols=True)
        
        for i, (waveforms, lengths) in enumerate(pbar):
            waveforms = waveforms.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_cuda):
                loss, _, _ = w_jepa_model(waveforms, lengths)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            # loss.item()은 스케일링되지 않은 값을 반환
            epoch_loss_meter.update(loss.item() * accumulation_steps, n=waveforms.size(0))

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            pbar.set_postfix(loss=f"{epoch_loss_meter.val:.4f}", avg_loss=f"{epoch_loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            global_step += 1
    
        avg_train_loss = epoch_loss_meter.avg
        all_epoch_train_losses.append(avg_train_loss)
        print(f"--- Epoch {epoch+1} Training Finished, Average Loss: {avg_train_loss:.4f} ---")

        # --- [추가] Validation Phase ---
        w_jepa_model.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", dynamic_ncols=True)
            for waveforms, lengths in val_pbar:
                waveforms = waveforms.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_cuda):
                    loss, _, _ = w_jepa_model(waveforms, lengths)
                
                val_loss_meter.update(loss.item(), n=waveforms.size(0))
                val_pbar.set_postfix(val_loss=f"{val_loss_meter.avg:.4f}")

        avg_val_loss = val_loss_meter.avg
        all_epoch_val_losses.append(avg_val_loss)
        print(f"--- Epoch {epoch+1} Validation Finished, Average Loss: {avg_val_loss:.4f} ---")


    print("--- End Training Loop ---")
    
    model_save_path = 'w_jepa_librispeech_bf16_accum_with_val.pth'
    torch.save(w_jepa_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    log_save_path = 'training_log_with_val.yaml'
    print(f"Saving training log to {log_save_path}")
    log_data = {
        'epoch_average_train_losses': all_epoch_train_losses,
        'epoch_average_val_losses': all_epoch_val_losses
        }
    with open(log_save_path, 'w') as f:
        yaml.dump(log_data, f, default_flow_style=False)
    print("Log saved.")


if __name__ == '__main__':
    main()