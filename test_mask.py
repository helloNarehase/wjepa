import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from typing import List, Tuple, Optional, Callable
import os
from pathlib import Path
from models import Feature_Extractor, Predictor, apply_masks, create_span_targets

# ====== Dataset Definition ======
class AudioDataset(Dataset):
    def __init__(self, audio_dir: str, file_ext: str = "*.wav"):
        """
        Simple audio dataset
        """
        self.audio_files = list(Path(audio_dir).glob(file_ext))
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        return waveform, sr, None, None, None, None


# ====== Enhanced Collate Function ======
def collate_fn(batch, sample_rate=16000, duration=10, min_duration_ms=200):
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


# ====== MaskCollator (Refactored) ======
class MaskCollator(object):
    def __init__(
        self,
        conv_configs:List[Tuple[int, int, int]],
        default_collate: Callable = collate_fn,
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
            return None, None, None
        
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
            
            # Ensure exact number of masked tokens
            if len(mask_indices) > total_mask_tokens:
                mask_indices = set(random.sample(list(mask_indices), total_mask_tokens))
            
            result.append(sorted(list(mask_indices)))
        return result
    
    def _extract_spans_from_mask(self, mask_indices: List[int], span_size: int) -> List[List[int]]:
        if not mask_indices:
            return []
        
        # Find consecutive groups of indices
        consecutive_groups = []
        current_group = [mask_indices[0]]
        for i in range(1, len(mask_indices)):
            if mask_indices[i] == mask_indices[i-1] + 1:
                current_group.append(mask_indices[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [mask_indices[i]]
        consecutive_groups.append(current_group)
        
        # Extract full spans from these groups
        spans = []
        for group in consecutive_groups:
            for i in range(0, len(group) - span_size + 1, span_size):
                spans.append(group[i:i+span_size])
        return spans


# ====== Feature Extractor ======
class MelSpectrogramExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, 
            hop_length=hop_length, power=2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def forward(self, waveforms):
        # waveforms: (B, 1, T)
        mel_spec = self.mel_transform(waveforms)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db


# ====== Masking Processor (Refactored) ======
# class MaskingProcessor:



# ====== Model Integration Example ======
class SimpleAudioModel(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=256, num_layers=6):
        super().__init__()
        self.feature_projection = nn.Linear(n_mels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, n_mels)
    
    def forward(self, unmasked_features, attention_mask=None):
        # unmasked_features: (B, n_mels, T) -> (B, T, n_mels)
        x = unmasked_features.transpose(1, 2)
        x = self.feature_projection(x)  # (B, T, hidden_dim)
        
        # Apply transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        output = self.output_projection(x)  # (B, T, n_mels)
        return output.transpose(1, 2)  # (B, n_mels, T)


# ====== Complete Training Pipeline Demo ======
def create_sample_dataset():
    """Create a dummy dataset for demonstration"""
    class DummyAudioDataset(Dataset):
        def __init__(self, num_samples=100): self.num_samples = num_samples
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            duration = random.uniform(2.0, 8.0)
            samples = int(16000 * duration)
            waveform = torch.randn(1, samples)
            return waveform, 16000, None, None, None, None

    return DummyAudioDataset()


def main():
    print("=== Refactored Audio Masking Pipeline Demo ===\n")
    
    # 1. Create dataset and dataloader
    dataset = create_sample_dataset()

    conv_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (512, 2, 2)
    ]

    SPAN = 4
    mask_collator = MaskCollator(conv_configs, span=SPAN, mask_ratio=0.65, sample_rate=16000)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=mask_collator)
    
    feature_extractor = Feature_Extractor(conv_configs=conv_configs)
    predictor         = Predictor(512, 512, 1, 32, 3.0, 1024, 0)
    
    # 3. Process a batch
    print("Processing one batch...")
    batch_data = next(iter(dataloader))
    
    if batch_data is None or batch_data[0] is None:
        print("Skipping empty batch.")
        return
        
    waveforms, lengths, ctx_mask, tgt_mask = batch_data
    
    print(f"  Waveforms shape: {waveforms.shape} | {lengths}")
    
    # Extract features from original audio
    with torch.no_grad():
        features, new_lengths = feature_extractor(waveforms, lengths)
        print(f"  Original features shape: {features.shape} | {new_lengths} | {lengths}")

    span_targets = create_span_targets(features, tgt_mask)

    ctx_pred = predictor(
        feature = features, 
        lengths = new_lengths, 
        ctx_mask = ctx_mask, 
        tgt_mask = tgt_mask,
    )

    B, M, S, D = span_targets.shape
    
    print(ctx_pred.shape, span_targets.reshape(B * M, S, D).shape)

    loss = F.l1_loss(ctx_pred, span_targets.reshape(B * M, S, D))
    print(loss)
    return

if __name__ == "__main__":
    main()
