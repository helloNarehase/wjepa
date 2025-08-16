from models import Predictor, Feature_Extractor
from utils import MaskCollator, AudioDataset, apply_masks
from torch.utils.data import DataLoader

if __name__ in "__main__":
    dataset = AudioDataset()

    mask_collator = MaskCollator(
        seq_length=1024,
        patch_size=8,
        enc_mask_scale=(0.4, 0.8),
        pred_mask_scale=(0.2, 0.2),
        enc_span_scale=(5, 10),
        pred_span_scale=(4, 4),
        nenc=1,
        npred=4,
        min_keep=1,
        allow_overlap=True,
        mask_strategy='contiguous_blocks'
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=mask_collator,
        num_workers=4,
    )

    conv_configs = [
        (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), 
        (512, 3, 2), (512, 2, 2), (512, 2, 2)
    ]

    model = Feature_Extractor(conv_configs)

    for (waveforms, lengths), en_mask, ta_mask in train_loader:
        print(waveforms.shape, lengths)
        features, new_lengths = model(waveforms, lengths)
        
        print(en_mask.shape, ta_mask.shape)
        print(en_mask.max(), ta_mask.max())
        print(features.shape)

        apply_masks(features.permute(0, 2, 1), en_mask)

        