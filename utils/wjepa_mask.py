import math
from multiprocessing import Value
from logging import getLogger
import torch
import torchaudio
from torch.nn import functional as F

GLOBAL_SEED = 0
logger = getLogger()

def collate_fn(batch, sample_rate=16000, duration=10, min_duration_ms=200):
    """
    Default collate function that processes a batch of audio data.
    It pads or truncates waveforms to a target length.
    """
    target_len = sample_rate * duration
    min_len = sample_rate * min_duration_ms // 1000
    waveforms, lengths = [], []
    
    for (waveform, sr, *_) in batch:
        # Skip waveforms that are too short
        if waveform.shape[1] < min_len:
            continue
        
        # Convert to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if the sample rate is different
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        current_len = waveform.shape[1]
        if current_len < target_len:
            # Pad shorter waveforms
            padding = target_len - current_len
            padded_waveform = F.pad(waveform, (0, padding))
            waveforms.append(padded_waveform)
            lengths.append(current_len)
        else:
            # Truncate longer waveforms from a random start
            start = torch.randint(0, current_len - target_len + 1, (1,)).item()
            truncated_waveform = waveform[:, start:start + target_len]
            waveforms.append(truncated_waveform)
            lengths.append(target_len)
    
    # Return None if the batch is empty after filtering
    if not waveforms:
        return None, None
    
    waveforms_tensor = torch.cat(waveforms, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return waveforms_tensor, lengths_tensor


class MaskCollator(object):
    def __init__(
        self,
        seq_length=1024,
        patch_size=16,
        enc_mask_scale=(0.15, 0.3),  # 더 보수적인 마스킹 비율
        pred_mask_scale=(0.15, 0.3),
        enc_span_scale=(1, 3),  # 스팬 길이 범위 조정
        pred_span_scale=(1, 3),
        nenc=1,
        npred=2,
        min_keep=1,  # 최소 유지 패치 수 감소
        allow_overlap=True,  # 겹침 허용으로 변경
        default_collate=collate_fn,
        mask_strategy='random_segments'
    ):
        """
        시계열 데이터용 마스크 콜레이터 - 재설계 버전
        """
        super(MaskCollator, self).__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.enc_span_scale = enc_span_scale
        self.pred_span_scale = pred_span_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.mask_strategy = mask_strategy
        self._itr_counter = Value('i', -1)
        self.default_collate = default_collate

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the output lengths of a convolutional feature extractor.
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size).float() / stride).long() + 1
        
        try:
            for module in self.conv_layers:
                conv = module[0]
                input_lengths = _conv_out_length(input_lengths, conv.kernel_size[0], conv.stride[0])
        except AttributeError:
            # conv_layers가 정의되지 않은 경우 원본 길이 반환
            pass
        return input_lengths

    def step(self):
        """Atomically increments and returns a shared counter."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _calculate_mask_params(self, valid_length, mask_scale, span_scale):
        """
        주어진 유효 길이에 대해 실제 적용 가능한 마스킹 파라미터 계산
        """
        # 마스킹할 패치 수 계산
        min_scale, max_scale = mask_scale
        scale = min_scale + torch.rand(1).item() * (max_scale - min_scale)
        target_mask_count = max(1, int(valid_length * scale))
        target_mask_count = min(target_mask_count, valid_length - self.min_keep)
        
        # 스팬 길이 계산
        min_span, max_span = span_scale
        span_length = min_span + int(torch.rand(1).item() * (max_span - min_span))
        span_length = min(span_length, valid_length)  # 유효 길이를 넘지 않도록
        
        return max(0, target_mask_count), max(1, span_length)

    def _generate_mask_indices(self, valid_length, target_mask_count, span_length):
        """
        실제 마스크 인덱스 생성 - 단순하고 안정적인 방법
        """
        if target_mask_count <= 0 or valid_length <= 0:
            return torch.tensor([], dtype=torch.long)
        
        if target_mask_count >= valid_length:
            return torch.arange(valid_length, dtype=torch.long)
        
        # 간단한 랜덤 샘플링 방식
        if self.mask_strategy == 'random_segments':
            # 연속된 세그먼트들을 랜덤하게 선택
            masked_indices = set()
            attempts = 0
            max_attempts = 100
            
            while len(masked_indices) < target_mask_count and attempts < max_attempts:
                # 랜덤 시작 위치 선택
                max_start = max(0, valid_length - span_length)
                if max_start <= 0:
                    start_pos = 0
                else:
                    start_pos = torch.randint(0, max_start + 1, (1,)).item()
                
                # 현재 스팬에서 추가할 수 있는 최대 길이
                remaining_count = target_mask_count - len(masked_indices)
                actual_span = min(span_length, remaining_count, valid_length - start_pos)
                
                # 인덱스 추가
                for i in range(actual_span):
                    if start_pos + i < valid_length:
                        masked_indices.add(start_pos + i)
                
                attempts += 1
            
            return torch.tensor(sorted(list(masked_indices)), dtype=torch.long)
        
        else:  # contiguous_blocks
            # 하나의 연속된 블록 생성
            if target_mask_count >= valid_length:
                return torch.arange(valid_length, dtype=torch.long)
            
            max_start = max(0, valid_length - target_mask_count)
            if max_start <= 0:
                start_pos = 0
            else:
                start_pos = torch.randint(0, max_start + 1, (1,)).item()
            
            end_pos = min(start_pos + target_mask_count, valid_length)
            return torch.arange(start_pos, end_pos, dtype=torch.long)

    def _create_mask(self, valid_length, mask_scale, span_scale, forbidden_regions=None):
        """
        단일 마스크 생성
        """
        target_mask_count, span_length = self._calculate_mask_params(valid_length, mask_scale, span_scale)
        
        # 금지된 영역이 있는 경우 고려 (겹침 방지)
        if forbidden_regions is not None and not self.allow_overlap:
            # 사용 가능한 위치만 고려
            available_positions = torch.ones(valid_length, dtype=torch.bool)
            for forbidden in forbidden_regions:
                if len(forbidden) > 0:
                    valid_forbidden = forbidden[forbidden < valid_length]
                    if len(valid_forbidden) > 0:
                        available_positions[valid_forbidden] = False
            
            available_indices = torch.where(available_positions)[0]
            if len(available_indices) < self.min_keep:
                # 사용 가능한 위치가 너무 적으면 겹침 허용
                mask_indices = self._generate_mask_indices(valid_length, target_mask_count, span_length)
            else:
                # 사용 가능한 위치에서만 마스킹
                available_length = len(available_indices)
                adjusted_count = min(target_mask_count, available_length - self.min_keep)
                if adjusted_count <= 0:
                    return torch.tensor([], dtype=torch.long)
                
                temp_indices = self._generate_mask_indices(available_length, adjusted_count, span_length)
                mask_indices = available_indices[temp_indices] if len(temp_indices) > 0 else torch.tensor([], dtype=torch.long)
        else:
            mask_indices = self._generate_mask_indices(valid_length, target_mask_count, span_length)
        
        return mask_indices

    def _patch_to_sequence_indices(self, patch_indices, seq_length):
        """
        패치 인덱스를 시퀀스 인덱스로 변환 (경계 체크 포함)
        """
        if len(patch_indices) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # 유효한 패치 인덱스만 필터링
        valid_patches = patch_indices[patch_indices >= 0]
        max_patch_idx = seq_length // self.patch_size
        valid_patches = valid_patches[valid_patches < max_patch_idx]
        
        if len(valid_patches) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # 패치를 시퀀스 인덱스로 확장
        expanded = valid_patches.unsqueeze(1) * self.patch_size
        offsets = torch.arange(self.patch_size)
        sequence_indices = (expanded + offsets).flatten()
        
        # 시퀀스 길이 내의 인덱스만 반환
        sequence_indices = sequence_indices[sequence_indices < seq_length]
        return sequence_indices

    def __call__(self, batch):
        """
        배치를 콜레이팅할 때 인코더와 예측기 마스크 생성
        """
        collated_batch = self.default_collate(batch)
        waveforms, wave_lengths = collated_batch
        
        if waveforms is None:
            return (None, None), None, None
        
        B = waveforms.shape[0]
        
        # 실제 유효한 패치 길이 계산
        try:
            feat_lengths = self._get_feat_extract_output_lengths(wave_lengths) -1
            valid_patch_lengths = [min(length.item() // self.patch_size, self.num_patches) 
                                 for length in feat_lengths]
        except (AttributeError, TypeError):
            # Fallback: 원본 웨이브폼 길이 기반
            valid_patch_lengths = [min(length.item() // self.patch_size, self.num_patches) 
                                 for length in wave_lengths]
        
        # 최소 길이 보장
        valid_patch_lengths = [max(1, length) for length in valid_patch_lengths]
        
        seed = self.step()
        torch.manual_seed(seed)
        
        batch_enc_masks = []
        batch_pred_masks = []
        
        for b_idx in range(B):
            valid_length = valid_patch_lengths[b_idx]
            
            # 예측기 마스크 생성
            pred_masks = []
            pred_patch_masks = []
            
            for _ in range(self.npred):
                pred_patch_indices = self._create_mask(valid_length, 
                                                     self.pred_mask_scale, 
                                                     self.pred_span_scale)
                pred_sequence_indices = self._patch_to_sequence_indices(pred_patch_indices, self.seq_length)
                pred_masks.append(pred_sequence_indices)
                pred_patch_masks.append(pred_patch_indices)
            
            # 인코더 마스크 생성
            enc_masks = []
            forbidden_regions = pred_patch_masks if not self.allow_overlap else None
            
            for _ in range(self.nenc):
                enc_patch_indices = self._create_mask(valid_length, 
                                                    self.enc_mask_scale, 
                                                    self.enc_span_scale,
                                                    forbidden_regions)
                enc_sequence_indices = self._patch_to_sequence_indices(enc_patch_indices, self.seq_length)
                enc_masks.append(enc_sequence_indices)
            
            batch_pred_masks.append(pred_masks)
            batch_enc_masks.append(enc_masks)
        
        # 배치를 위한 패딩 처리
        def pad_masks_to_same_length(batch_masks, fill_value=-1):
            """모든 마스크를 동일한 길이로 패딩"""
            if not batch_masks or not batch_masks[0]:
                return torch.tensor([]).reshape(0, 0, 0)
            
            # 최대 길이 찾기
            max_len = 0
            for b_masks in batch_masks:
                for mask in b_masks:
                    max_len = max(max_len, len(mask))
            
            if max_len == 0:
                # 모든 마스크가 비어있는 경우
                B = len(batch_masks)
                num_masks = len(batch_masks[0])
                return torch.full((B, num_masks, 1), fill_value, dtype=torch.long)
            
            # 패딩 적용
            padded_batch = []
            for b_masks in batch_masks:
                padded_masks = []
                for mask in b_masks:
                    if len(mask) < max_len:
                        padding = torch.full((max_len - len(mask),), fill_value, dtype=torch.long)
                        padded_mask = torch.cat([mask, padding])
                    else:
                        padded_mask = mask[:max_len]  # 혹시나 하는 안전 장치
                    padded_masks.append(padded_mask)
                padded_batch.append(torch.stack(padded_masks))
            
            return torch.stack(padded_batch)
        
        final_enc_masks = pad_masks_to_same_length(batch_enc_masks)
        final_pred_masks = pad_masks_to_same_length(batch_pred_masks)
        
        return collated_batch, final_enc_masks, final_pred_masks

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


# 사용 예시
if __name__ == "__main__":

    # import torch

    # x = torch.tensor([[0], [1], [2]])
    # N = 2
    # dim = 0

    # out = torch.repeat_interleave(x, N, dim=dim)
    # print(out)


    # exit()
    # 길이가 다른 시계열 데이터 예시
    sample_batch = [
        torch.randn(1024, 1024),
        torch.randn(1024, 1024),
        torch.randn(1024, 1024),
    ]
    
    # 실제 시퀀스 길이들 (패딩 제외)
    sequence_lengths = [800, 512, 512]
    
    # 마스크 콜레이터 초기화
    mask_collator = MaskCollator(
        seq_length=1024,
        patch_size=16,
        enc_mask_scale=(0.4, 0.8),
        pred_mask_scale=(0.2, 0.2),
        enc_span_scale=(5, 10),
        pred_span_scale=(4, 4),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
        mask_strategy='contiguous_blocks'
    )
    
    # 마스크 생성
    batch, enc_masks, pred_masks = mask_collator(sample_batch, lengths=sequence_lengths)
    
    print(f"Original batch shape: {batch.shape}")
    print(f"Encoder masks shape: {enc_masks.shape}")
    print(f"Predictor masks shape: {pred_masks.shape}")
    print("-" * 50)

    # --- 테스트 케이스 1: 2D 마스크 적용 (B, M) -> (B, L', D) ---
    # enc_masks (B, N, M)에서 첫 번째 마스크 세트만 선택 -> (B, M)
    single_enc_mask = enc_masks[:, 0, :]
    print(f"Applying single encoder mask with shape: {single_enc_mask.shape}")
    masked_output_1 = apply_masks(batch, single_enc_mask, patch_size=16)
    print(f"  -> Output shape: {masked_output_1.shape}")
    print("-" * 50)

    # --- 테스트 케이스 2: 3D 인코더 마스크 적용 (B, N, M) -> (N, B, L', D) ---
    print(f"Applying all encoder masks with shape: {enc_masks.shape}")
    masked_output_2 = apply_masks(batch, enc_masks, patch_size=16)
    print(f"  -> Output shape: {masked_output_2.shape}")
    print("-" * 50)

    # --- 테스트 케이스 3: 3D 예측기 마스크 적용 (B, N, M) -> (N, B, L', D) ---
    print(f"Applying all predictor masks with shape: {pred_masks.shape}")
    masked_output_3 = apply_masks(batch, pred_masks, patch_size=16)
    print(f"  -> Output shape: {masked_output_3.shape}")
    print("-" * 50)

