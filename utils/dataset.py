from torch.utils.data import Dataset
from torchaudio.functional import resample
from torchcodec.decoders import AudioDecoder

from os import listdir
from os.path import join

class AudioDataset(Dataset):
    def __init__(self, root_dir='./ganyu', target_sampling_rate=16000):
        self.root_dir = root_dir
        self.target_sampling_rate = target_sampling_rate
        self.filepaths = [
            join(root_dir, filename)
            for filename in listdir(root_dir)
            if filename.endswith(".wav")
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        decoder = AudioDecoder(filepath)
        audio_tensor, _, _, sampling_rate = decoder.get_all_samples()
        """
        example:
            AudioSamples:
                data (shape): torch.Size([2, 4297722])
                pts_seconds: 0.02505668934240363
                duration_seconds: 97.45401360544217
                sample_rate: 44100
        """

        if sampling_rate != self.target_sampling_rate:
            audio_tensor = resample(
                audio_tensor,
                orig_freq=sampling_rate,
                new_freq=self.target_sampling_rate
            )

        return audio_tensor, sampling_rate, None, None, None, None