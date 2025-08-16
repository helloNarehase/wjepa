import torch
from torch import nn
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
from typing import List, Tuple, Optional, Callable, Dict
from collections import OrderedDict

from torchcodec.decoders import AudioDecoder
from torchaudio.functional import resample
from os import listdir
from os.path import join, splitext
from pathlib import Path

from torch.amp import autocast, GradScaler
# Assuming 'models.py' contains the WaveEncode definition from your original setup
from models import WaveEncode 


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

# --- Tokenizer (Jaso-level for Korean) ---
# 한글 자모 리스트
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 자소 단위 전체 문자 집합 생성
KOREAN_JASO_CHARS = "".join(sorted(list(set(CHOSUNG_LIST + JUNGSUNG_LIST + JONGSUNG_LIST))))
# 기타 필요한 문자 추가
OTHER_CHARS = " ',-.?!0123456789abcdefghijklmnopqrstuvwxyz"
JASO_VOCAB = KOREAN_JASO_CHARS + OTHER_CHARS


class Tokenizer:
    """CTC를 위한 간단한 한국어 자소(Jaso) 단위 토크나이저."""
    
    def __init__(self, characters: str = JASO_VOCAB):
        self.blank_char = '<blank>'
        self.unk_char = '<unk>'
        
        self.char_map = {self.blank_char: 0}
        self.index_map = {0: self.blank_char}
        
        for i, char in enumerate(characters, start=1):
            if char == '': continue # 종성 리스트의 빈 문자열은 제외
            self.char_map[char] = i
            self.index_map[i] = char
            
        self.unk_char_idx = len(self.char_map)
        self.char_map[self.unk_char] = self.unk_char_idx
        self.index_map[self.unk_char_idx] = self.unk_char

    def _decompose(self, text: str) -> List[str]:
        """문자열을 자소 단위로 분해합니다."""
        decomposed = []
        for char in text:
            if '가' <= char <= '힣':
                char_code = ord(char) - ord('가')
                
                choseong_idx = char_code // (21 * 28)
                decomposed.append(CHOSUNG_LIST[choseong_idx])
                
                jungsung_idx = (char_code % (21 * 28)) // 28
                decomposed.append(JUNGSUNG_LIST[jungsung_idx])
                
                jongseong_idx = char_code % 28
                if jongseong_idx > 0:
                    decomposed.append(JONGSUNG_LIST[jongseong_idx])
            else:
                decomposed.append(char)
        return decomposed

    def text_to_indices(self, text: str) -> List[int]:
        """문자열을 자소 단위로 분해하여 정수 인덱스 리스트로 변환합니다."""
        decomposed_text = self._decompose(text.lower())
        return [self.char_map.get(jamo, self.unk_char_idx) for jamo in decomposed_text]

    def indices_to_text(self, indices: List[int]) -> str:
        """정수 인덱스 리스트를 자소 문자열로 변환합니다."""
        return "".join([self.index_map.get(i, self.unk_char) for i in indices])

    def get_vocab_size(self) -> int:
        """어휘 사전에 있는 총 토큰의 개수를 반환합니다."""
        return len(self.char_map)
        
# --- 데이터 처리 및 학습 루프 ---

class AudioTextDataset(Dataset):
    """
    Dataset for loading audio files and their corresponding transcriptions.
    """
    def __init__(self, root_dir='./ganyu', tokenizer: Tokenizer = None, target_sampling_rate=16000, min_duration_sec=1.0, max_duration_sec=20.0):
        self.root_dir = root_dir
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.target_sampling_rate = target_sampling_rate
        
        all_filepaths = {
            splitext(filename)[0]: join(root_dir, filename)
            for filename in listdir(root_dir)
            if filename.endswith(".wav")
        }
        
        self.data_pairs = []
        print("Scanning dataset, loading transcripts, and filtering by duration...")
        for file_id, audio_path in tqdm(all_filepaths.items(), desc="Processing files"):
            c = splitext(audio_path)[0].split("_")[0]
            transcription_path = f"{c}_transcription.txt"
            try:
                # Check audio duration first
                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
                if not (min_duration_sec <= duration <= max_duration_sec):
                    continue

                # Read transcription
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
                
                if transcription: # Ensure transcription is not empty
                    self.data_pairs.append((audio_path, transcription))

            except FileNotFoundError:
                print(f"Warning: Transcription not found for {audio_path}, skipping.")
            except Exception as e:
                print(f"Warning: Could not process {audio_path}, skipping. Error: {e}")

        print(f"Finished processing. Found {len(self.data_pairs)} valid audio-text pairs.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        audio_path, transcription = self.data_pairs[idx]
        
        try:
            codec = AudioDecoder(audio_path)
            waveform, _, _, sample_rate = codec.get_all_samples()
        except Exception as e:
            print(f"Warning: Error loading file {audio_path}: {e}")
            return None, None, None # Return transcription for WER calculation

        if sample_rate != self.target_sampling_rate:
            waveform = resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.target_sampling_rate
            )
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze()
        
        encoded_transcription = torch.tensor(self.tokenizer.text_to_indices(transcription), dtype=torch.long)
        
        # Return original transcription for WER calculation
        return waveform, encoded_transcription, transcription

class CTCCollateFn:
    """
    Collator for CTC training. Pads audio and text sequences.
    """
    def __init__(self, conv_configs: List[Tuple[int, int, int]]):
        self.conv_configs = conv_configs

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculates the output sequence length of the convolutional feature extractor."""
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size).float() / stride).long() + 1
        
        lengths = input_lengths.clone()
        for conv in self.conv_configs:
            lengths = _conv_out_length(lengths, conv[1], conv[2])
        return lengths

    def __call__(self, batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return None, None, None, None, None, None

        waveforms, transcripts_encoded, transcripts_raw = zip(*batch)
        
        waveform_lengths = torch.tensor([w.size(0) for w in waveforms])
        padded_waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        
        transcript_lengths = torch.tensor([len(t) for t in transcripts_encoded])
        padded_transcripts = nn.utils.rnn.pad_sequence(transcripts_encoded, batch_first=True)
        
        encoder_output_lengths = self._get_feat_extract_output_lengths(waveform_lengths)

        return padded_waveforms, waveform_lengths, padded_transcripts, transcript_lengths, encoder_output_lengths, transcripts_raw


# --- 모델 정의 ---
class ASRModel(nn.Module):
    """
    Acoustic model for CTC-based Speech Recognition.
    """
    def __init__(self, encoder: WaveEncode, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classification_head = nn.Linear(encoder.blocks.dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        features, new_lengths = self.encoder.feature_extractor(x.unsqueeze(1), lengths)
        encoded_features = self.encoder.encode(features.permute(0, 2, 1), new_lengths)
        logits = self.classification_head(encoded_features)
        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        return log_probs, new_lengths

def get_lr_scheduler(optimizer, max_iter, warmup_iter, final_lr):
    """Cosine decay learning rate scheduler with warmup."""
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

def calculate_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Calculates the Word Error Rate (WER) manually using Levenshtein distance.
    This function computes the corpus-level WER.
    """
    # Join all sentences into a single string and then split by space to get a list of words
    ref_words = " ".join(references).split()
    hyp_words = " ".join(hypotheses).split()

    # Get the number of words in the reference
    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 1.0 if m > 0 else 0.0

    # Initialize the DP table (Levenshtein distance)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if hyp_words[i - 1] == ref_words[j - 1] else 1
            
            deletion_cost = dp[i - 1][j] + 1
            insertion_cost = dp[i][j - 1] + 1
            substitution_cost = dp[i - 1][j - 1] + cost
            
            dp[i][j] = min(deletion_cost, insertion_cost, substitution_cost)

    errors = dp[m][n]
    wer = float(errors) / n
    return wer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f"Using device: {device}")

    batch_size = 16
    base_learning_rate = 1e-5
    final_learning_rate = 1e-12
    num_epochs = 600
    accumulation_steps = 8
    
    conv_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (512, 2, 2)
    ]
    encoder_dim = 512
    encoder_layers = 6
    encoder_heads = 8
    encoder_dropout = 0.2
    encoder_droppath = 0.1
    
    tokenizer = Tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    print("Loading dataset...")
    full_dataset = AudioTextDataset(tokenizer=tokenizer, min_duration_sec=1.0, max_duration_sec=15.0)

    if len(full_dataset) == 0:
        print("No data found. Exiting.")
        return

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    print(f"Splitting dataset: {train_size} for training, {val_size} for validation.")
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    ctc_collator = CTCCollateFn(conv_configs=conv_configs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ctc_collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ctc_collator, num_workers=4, pin_memory=True)
    
    encoder = WaveEncode(
        conv_configs=conv_configs, nlayers=encoder_layers, dim=encoder_dim, 
        nhead=encoder_heads, ratio=4.0, dropout=encoder_dropout, droppath=encoder_droppath
    )
    
    checkpoint = torch.load('w_jepa_online_network_final.pth', map_location=device)
    context_encoder_state_dict = OrderedDict()
    prefix = 'context_encoder.'
    for key, value in checkpoint.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            context_encoder_state_dict[new_key] = value
    encoder.load_state_dict(context_encoder_state_dict, strict=False)
    
    # print("--- Freezing Encoder Parameters ---")
    # for param in encoder.parameters():
    #     param.requires_grad = False

    model = ASRModel(encoder=encoder, num_classes=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=0.01)
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    
    max_iter = len(train_loader) // accumulation_steps * num_epochs
    scheduler = get_lr_scheduler(optimizer, max_iter, int(max_iter * 0.001), final_learning_rate)
    scaler = GradScaler(enabled=use_cuda)
    
    all_epoch_train_losses = []
    all_epoch_val_losses = []
    all_epoch_val_wers = [] # To store WER for each epoch

    print("--- Start Training Loop ---")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", dynamic_ncols=True)
        
        for i, batch_data in enumerate(pbar):
            if batch_data[0] is None: continue
            
            waveforms, waveform_lengths, transcripts, transcript_lengths, _, _ = batch_data
            
            waveforms = waveforms.to(device, non_blocking=True)
            waveform_lengths = waveform_lengths.to(device, non_blocking=True)
            transcripts = transcripts.to(device, non_blocking=True)
            transcript_lengths = transcript_lengths.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_cuda):
                log_probs, encoder_output_lengths = model(waveforms, waveform_lengths)
                loss = ctc_loss_fn(log_probs, transcripts, encoder_output_lengths, transcript_lengths)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            epoch_loss_meter.update(loss.item() * accumulation_steps, n=waveforms.size(0))

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            pbar.set_postfix(loss=f"{epoch_loss_meter.val:.4f}", avg_loss=f"{epoch_loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    
        avg_train_loss = epoch_loss_meter.avg
        all_epoch_train_losses.append(avg_train_loss)
        print(f"--- Epoch {epoch+1} Training Finished, Average Loss: {avg_train_loss:.4f} ---")

        # --- Validation Phase with WER Calculation ---
        model.eval()
        val_loss_meter = AverageMeter()
        all_refs = []
        all_hyps = []
        first_sample_logged_this_epoch = False # Flag to log only the first sample
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", dynamic_ncols=True)
            for batch_data in val_pbar:
                if batch_data[0] is None: continue

                waveforms, waveform_lengths, transcripts, transcript_lengths, _, transcripts_raw = batch_data
                waveforms = waveforms.to(device, non_blocking=True)
                waveform_lengths = waveform_lengths.to(device, non_blocking=True)
                transcripts = transcripts.to(device, non_blocking=True)
                transcript_lengths = transcript_lengths.to(device, non_blocking=True)
                
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_cuda):
                    log_probs, encoder_output_lengths = model(waveforms, waveform_lengths)
                    loss = ctc_loss_fn(log_probs, transcripts, encoder_output_lengths, transcript_lengths)
                
                val_loss_meter.update(loss.item(), n=waveforms.size(0))
                
                # --- Greedy Decoding for WER ---
                pred_indices = log_probs.argmax(dim=-1).T
                
                for i in range(pred_indices.size(0)):
                    pred_seq_len = encoder_output_lengths[i]
                    pred_seq = pred_indices[i, :pred_seq_len]
                    
                    decoded_indices = []
                    last_idx = -1
                    for idx in pred_seq:
                        if idx.item() != last_idx and idx.item() != 0:
                            decoded_indices.append(idx.item())
                        last_idx = idx.item()

                    hyp_text = tokenizer.indices_to_text(decoded_indices)
                    ref_text = "".join(tokenizer._decompose(transcripts_raw[i]))
                    
                    all_hyps.append(hyp_text)
                    all_refs.append(ref_text)

                    # --- Log the first sample of the epoch ---
                    if not first_sample_logged_this_epoch:
                        print(f"\n--- First Validation Sample (Epoch {epoch+1}) ---")
                        print(f"  Reference : {ref_text}")
                        print(f"  Hypothesis: {hyp_text}")
                        first_sample_logged_this_epoch = True

                val_pbar.set_postfix(val_loss=f"{val_loss_meter.avg:.4f}")
        
        # Calculate WER for the entire validation set using the manual function
        wer = calculate_wer(all_refs, all_hyps)
        all_epoch_val_wers.append(wer)

        avg_val_loss = val_loss_meter.avg
        all_epoch_val_losses.append(avg_val_loss)
        print(f"--- Epoch {epoch+1} Validation Finished ---")
        print(f"Average Loss: {avg_val_loss:.4f}")
        print(f"Word Error Rate (WER): {wer * 100:.2f}%")
        print("-----------------------------------------")

        model_save_path = f'asr_ctc_model_epoch_{epoch+1}.pth'
        # torch.save(model.state_dict(), model_save_path)
        # print(f"Model checkpoint saved to {model_save_path}")

    print("--- End Training Loop ---")
    
    model_save_path = 'asr_ctc_model_final.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    log_save_path = 'training_log_ctc.yaml'
    print(f"Saving training log to {log_save_path}")
    log_data = {
        'epoch_average_train_losses': all_epoch_train_losses,
        'epoch_average_val_losses': all_epoch_val_losses,
        'epoch_average_val_wers': all_epoch_val_wers
    }
    with open(log_save_path, 'w') as f:
        yaml.dump(log_data, f, default_flow_style=False)
    print("Log saved.")


if __name__ == '__main__':
    main()
