import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F


class Preprocess(nn.Module):
    def __init__(
        self,
        sample_rate=8000,
        n_fft=256,
        hop_length=128,
        n_mels=40,
    ):
        """
        Args:
            sample_rate: Target sample rate for audio
            n_fft: Number of FFT bins
            hop_length: Hop length for STFT
            n_mels: Number of Mel filterbanks
            max_len: Maximum waveform length in samples for padding/truncating
        """
        super().__init__()

        self.mel_spec = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            ),
            torchaudio.transforms.AmplitudeToDB()
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.float()

        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()

        mel_spec_db = self.mel_spec(waveform)

        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        return mel_spec_db


def collate_fn(batch: torch.Tensor, target_len=100):
    mels, labels = zip(*batch)

    padded_mels = []
    for m in mels:
        mel_length = m.shape[-1]
        if mel_length < target_len:
            m = F.pad(m, (0, target_len - mel_length))
        elif mel_length > target_len:
            m = m[:, :target_len]
        padded_mels.append(m)

    mels = torch.stack(padded_mels)
    mels = mels.unsqueeze(1)
    labels = torch.tensor(labels)

    return mels, labels


preprocess = Preprocess()