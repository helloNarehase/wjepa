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
import numpy as np
from typing import List, Tuple, Optional, Callable


from torchaudio import load
from torchaudio.functional import resample
from os import listdir
from os.path import join
from pathlib import Path

from torch.amp import autocast, GradScaler
from models import WaveEncode, Predictor, apply_masks, create_span_targets


# --- 특정 UserWarning 무시 ---
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec")

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

# --- 데이터 처리 및 학습 루프 ---

def base_collate_fn(batch, sample_rate=16000, duration=10, min_duration_ms=200):
    target_len = sample_rate * duration
    min_len = sample_rate * min_duration_ms // 1000
    waveforms, lengths = [], []
    
    for (waveform, sr, _, _, _, _) in batch:
        if waveform.shape[1] < min_len:
            continue

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
    
    if not waveforms:
        return None, None
            
    waveforms_tensor = torch.cat(waveforms, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return waveforms_tensor, lengths_tensor


# ====== MaskCollator (test_mask.py에서 가져와 수정) ======
class MaskCollator(object):
    def __init__(
        self,
        conv_configs:List[Tuple[int, int, int]],
        default_collate: Callable = base_collate_fn,
        span: int = 4,
        mask_ratio: float = 0.6,
        sample_rate: int = 16000,
        **collate_kwargs
    ):
        self.default_collate = default_collate
        self.span = span
        self.mask_ratio = mask_ratio
        self.sample_rate = sample_rate
        self.collate_kwargs = collate_kwargs
        self.conv_configs = conv_configs

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size).float() / stride).long() + 1
        for conv in self.conv_configs:
            input_lengths = _conv_out_length(input_lengths, conv[1], conv[2])
        return input_lengths

    def __call__(self, batch):
        """
        Processes a batch of audio, creates masks, and returns data in the specified format.
        
        Returns:
            Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
            - waveforms: (B, 1, T_audio)
            - full_mask: List of B tensors, each with indices of frames to be masked.
            - target_mask: (B, N, S) tensor with indices of spans for the reconstruction target.
        """
        result = self.default_collate(batch, **self.collate_kwargs)
        
        if result is None or result[0] is None:
            return None, None, None, None
        
        waveforms, lengths = result
        _lengths = self._get_feat_extract_output_lengths(lengths)
        frame_lengths = _lengths.tolist()
        
        full_masks_lists, extracted_spans_np = self.create_mask_indices_with_extraction(
            frame_lengths, self.span, self.mask_ratio
        )
        
        # Convert full_masks to a list of torch.Tensors
        full_mask = [torch.tensor(mask, dtype=torch.long) for mask in full_masks_lists]
        
        # Convert extracted_spans numpy array to a torch.Tensor
        if extracted_spans_np.size > 0:
            target_mask = torch.from_numpy(extracted_spans_np)
        else:
            # Create an empty tensor if no spans were extracted
            B = len(_lengths)
            target_mask = torch.empty((B, 0, self.span), dtype=torch.long)
            
        return waveforms, lengths, full_mask, target_mask
    
    def create_mask_indices_with_extraction(self, lengths: List[int], span: int, mask_ratio: float) -> Tuple[List[List[int]], np.ndarray]:
        if not lengths:
            return [], np.array([])
        
        full_masks = self._create_full_masks(lengths, span, mask_ratio)
        
        all_spans = []
        min_num_spans = float('inf')
        
        for mask_indices in full_masks:
            spans = self._extract_spans_from_mask(mask_indices, span)
            all_spans.append(spans)
            min_num_spans = min(min_num_spans, len(spans))
        
        if min_num_spans == float('inf'):
            min_num_spans = 0
        
        N = int(min_num_spans)
        B = len(lengths)
        
        if N == 0:
            return full_masks, np.array([])
        
        extracted_spans = np.full((B, N, span), -1, dtype=int)
        
        for b, spans in enumerate(all_spans):
            selected_spans = random.sample(spans, N) if len(spans) >= N else spans
            for n, span_indices in enumerate(selected_spans):
                for s, idx in enumerate(span_indices):
                    extracted_spans[b, n, s] = idx
        
        return full_masks, extracted_spans
    
    def _create_full_masks(self, lengths: List[int], span: int, mask_ratio: float) -> List[List[int]]:
        result = []
        for length in lengths:
            if length <= span:
                result.append([])
                continue
            
            total_mask_tokens = int(length * mask_ratio)
            mask_indices = set()
            
            max_attempts = length * 2
            for _ in range(max_attempts):
                start_pos = random.randint(0, length - span)
                new_indices = set(range(start_pos, start_pos + span))
                
                if not mask_indices.intersection(new_indices):
                    mask_indices.update(new_indices)
                
                if len(mask_indices) >= total_mask_tokens:
                    break
            
            if len(mask_indices) > total_mask_tokens:
                mask_indices = set(random.sample(list(mask_indices), total_mask_tokens))
            
            result.append(sorted(list(mask_indices)))
        return result
    
    def _extract_spans_from_mask(self, mask_indices: List[int], span_size: int) -> List[List[int]]:
        if not mask_indices:
            return []
        
        consecutive_groups = []
        current_group = [mask_indices[0]]
        for i in range(1, len(mask_indices)):
            if mask_indices[i] == mask_indices[i-1] + 1:
                current_group.append(mask_indices[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [mask_indices[i]]
        consecutive_groups.append(current_group)
        
        spans = []
        for group in consecutive_groups:
            for i in range(0, len(group) - span_size + 1, span_size):
                spans.append(group[i:i+span_size])
        return spans


# --- 모델 정의 ---
class W_JEPA(nn.Module):
    """
    W_JEPA의 온라인(Online) 네트워크.
    Context Encoder와 Predictor를 포함하며, 그래디언트 업데이트를 통해 학습됩니다.
    Target Encoder는 이 클래스 외부에 존재하며 EMA로 업데이트됩니다.
    """
    def __init__(
        self,
        encoder: WaveEncode,
        predictor: Predictor,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.predictor = predictor

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, 
                ctx_mask: List[torch.Tensor], 
                tgt_mask: torch.Tensor,
                target_encoder: nn.Module):
        
        # 1. Feature Extraction
        features, new_lengths = self.context_encoder.feature_extractor(x, lengths)
        
        # 2. Context Encoding
        ctx_feature, _lengths = apply_masks(features, ctx_mask)
        encoded_context = self.context_encoder.encode(ctx_feature.permute(0, 2, 1), _lengths)

        # 3. Target Encoding
        with torch.no_grad():
            target_encoder.eval()
            full_encoded_features = target_encoder.encode(features.permute(0, 2, 1).detach(), lengths=new_lengths.to(x.device))
            tgt_features = create_span_targets(full_encoded_features.permute(0, 2, 1), tgt_mask)

        # If there are no targets, we can't compute a loss. Return 0.
        if tgt_features is None:
            return torch.tensor(0., device=x.device, requires_grad=True)

        # 4. Prediction
        ctx_pred = self.predictor(
            ctx_feature=encoded_context.permute(0, 2, 1),
            length=features.size(2),
            ctx_mask=ctx_mask,
            tgt_mask=tgt_mask
        )

        # 5. Loss Calculation
        tgt_features = tgt_features.detach()
        B, M, S, D = tgt_features.shape
        loss = F.l1_loss(ctx_pred, tgt_features.reshape(B * M, S, D))
        
        return loss

class AudioDataset(Dataset):
    def __init__(self, root_dir='./ganyu', target_sampling_rate=16000, min_duration_sec=5.0):
        self.root_dir = root_dir
        self.target_sampling_rate = target_sampling_rate
        
        all_filepaths = [
            join(root_dir, filename)
            for filename in listdir(root_dir)
            if filename.endswith(".wav")
        ]
        
        self.filepaths = []
        print("Scanning dataset and filtering by duration...")
        for filepath in tqdm(all_filepaths, desc="Filtering audio files"):
            try:
                info = torchaudio.info(filepath)
                duration = info.num_frames / info.sample_rate
                if duration >= min_duration_sec:
                    self.filepaths.append(filepath)
            except Exception as e:
                print(f"Warning: Could not read info for {filepath}, skipping. Error: {e}")

        print(f"Finished filtering. Found {len(all_filepaths)} total files, kept {len(self.filepaths)} files >= {min_duration_sec}s.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        try:
            audio_tensor, sampling_rate = load(filepath)
        except Exception as e:
            print(f"Warning: Error loading file {filepath}: {e}")
            # Return a dummy tensor that will be filtered by collate_fn
            return torch.zeros(1, 1), self.target_sampling_rate, None, None, None, None

        if sampling_rate != self.target_sampling_rate:
            audio_tensor = resample(
                audio_tensor,
                orig_freq=sampling_rate,
                new_freq=self.target_sampling_rate
            )

        return audio_tensor, sampling_rate, None, None, None, None

def get_lr_scheduler(optimizer, max_iter, warmup_iter, final_lr):
    base_lr = optimizer.param_groups[0]['lr']
    def lr_lambda(current_iter):
        if current_iter < warmup_iter:
            return float(current_iter) / float(max(1, warmup_iter))
        elif current_iter >= max_iter:
            return final_lr / base_lr
        else:
            progress = float(current_iter - warmup_iter) / float(max(1, max_iter - warmup_iter))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            scaled_lr = final_lr + (base_lr - final_lr) * cosine_decay
            return scaled_lr / base_lr
    return LambdaLR(optimizer, lr_lambda)

def main():
    # --- 학습 하이퍼파라미터 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f"Using device: {device}")

    batch_size = 32
    base_learning_rate = 3e-4
    final_learning_rate = 1e-9
    num_epochs = 30
    accumulation_steps = 8
    ema_decay = 0.996

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
    
    max_seq_len = 10_000

    # --- 데이터셋 및 DataLoader 준비 ---
    print("Loading dataset...")
    full_dataset = AudioDataset(min_duration_sec=5.0)

    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    print(f"Splitting dataset: {train_size} for training, {val_size} for validation.")
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    mask_collator = MaskCollator(
        conv_configs=conv_configs,
        span=4,
        mask_ratio=0.65,
        sample_rate=16000
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mask_collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mask_collator, num_workers=4, pin_memory=True)
    
    # --- 모델 초기화 ---
    # 1. 온라인 네트워크(Context Encoder)와 Predictor 초기화
    context_encoder = WaveEncode(
        conv_configs=conv_configs, nlayers=encoder_layers, dim=encoder_dim, 
        nhead=encoder_heads, ratio=4.0, dropout=encoder_dropout, droppath=encoder_droppath
    ).to(device)

    predictor = Predictor(
        in_dim=encoder_dim, dim=predictor_dim, nlayers=predictor_layers, 
        nhead=predictor_heads, ratio=4.0, max_seq_len=max_seq_len, droppath=predictor_droppath
    ).to(device)

    # W_JEPA 모델은 온라인 네트워크(Context Encoder)와 Predictor를 래핑합니다.
    w_jepa_model = W_JEPA(encoder=context_encoder, predictor=predictor).to(device)

    # 2. 타겟 네트워크(Target Encoder)를 온라인 네트워크의 복사본으로 초기화
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # 3. EMA 업데이터 초기화
    ema_updater = EMA(beta=ema_decay)
    
    # 옵티마이저는 온라인 네트워크의 파라미터만 최적화합니다.
    optimizer = torch.optim.AdamW(w_jepa_model.parameters(), lr=base_learning_rate, weight_decay=0.02)
    
    max_iter = len(train_loader) // accumulation_steps * num_epochs
    scheduler = get_lr_scheduler(optimizer, max_iter, int(max_iter * 0.3), final_learning_rate)
    
    scaler = GradScaler(enabled=use_cuda)
    
    all_epoch_train_losses = []
    all_epoch_val_losses = []

    # --- 학습 루프 ---
    print("--- Start Training Loop ---")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        w_jepa_model.train()
        epoch_loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", dynamic_ncols=True)
        
        for i, batch_data in enumerate(pbar):
            if batch_data[0] is None: continue
            
            waveforms, lengths, ctx_mask, tgt_mask = batch_data
            waveforms = waveforms.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            ctx_mask = [m.to(device, non_blocking=True) for m in ctx_mask]
            tgt_mask = tgt_mask.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_cuda):
                # Target Encoder는 forward pass에 인자로 전달됩니다.
                loss = w_jepa_model(waveforms, lengths, ctx_mask, tgt_mask, target_encoder)
                if loss.requires_grad:
                    loss = loss / accumulation_steps
            
            if loss.requires_grad:
                scaler.scale(loss).backward()
                epoch_loss_meter.update(loss.item() * accumulation_steps, n=waveforms.size(0))

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                # EMA 업데이트를 명시적으로 호출합니다.
                update_moving_average(ema_updater, target_encoder, w_jepa_model.context_encoder)
                optimizer.zero_grad()
                scheduler.step()
            
            pbar.set_postfix(loss=f"{epoch_loss_meter.val:.4f}", avg_loss=f"{epoch_loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    
        avg_train_loss = epoch_loss_meter.avg
        all_epoch_train_losses.append(avg_train_loss)
        print(f"--- Epoch {epoch+1} Training Finished, Average Loss: {avg_train_loss:.4f} ---")

        # --- Validation Phase ---
        w_jepa_model.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", dynamic_ncols=True)
            for batch_data in val_pbar:
                if batch_data[0] is None: continue

                waveforms, lengths, ctx_mask, tgt_mask = batch_data
                waveforms = waveforms.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                ctx_mask = [m.to(device, non_blocking=True) for m in ctx_mask]
                tgt_mask = tgt_mask.to(device, non_blocking=True)
                
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_cuda):
                    loss = w_jepa_model(waveforms, lengths, ctx_mask, tgt_mask, target_encoder)
                
                val_loss_meter.update(loss.item(), n=waveforms.size(0))
                val_pbar.set_postfix(val_loss=f"{val_loss_meter.avg:.4f}")

        avg_val_loss = val_loss_meter.avg
        all_epoch_val_losses.append(avg_val_loss)
        print(f"--- Epoch {epoch+1} Validation Finished, Average Loss: {avg_val_loss:.4f} ---")

    print("--- End Training Loop ---")
    
    # 모델 저장 시 온라인 네트워크(w_jepa_model)만 저장해도 충분합니다.
    # 필요 시 target_encoder도 별도로 저장할 수 있습니다.
    model_save_path = 'w_jepa_online_network_final.pth'
    torch.save(w_jepa_model.state_dict(), model_save_path)
    print(f"Online model saved to {model_save_path}")

    log_save_path = 'training_log_refactored.yaml'
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
