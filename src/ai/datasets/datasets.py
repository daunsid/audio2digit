import torch
import torchaudio
import io
import librosa
from torch.nn import functional as F
from torch.utils.data import Dataset
from ai.preprocessing import preprocess


class SpokenDigitDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, target_transform=None):
        """
        Args:
            hf_dataset: Hugging Face dataset split (train/test).
            transform: function/transform applied to audio waveform (e.g., mel spectrogram).
            target_transform: optional transform for labels.
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[int(idx)]
        waveform = torch.tensor(sample["audio"]["array"]).float()
        label = sample["label"]

        if self.transform:
            waveform = self.transform(waveform)

        if self.target_transform:
            label = self.target_transform(label)

        return waveform, torch.tensor(label, dtype=torch.long)


def resample(waveform, orig_sr, target_sr=8000) -> torch.Tensor:
    """
    Resample waveform if its sampling rate != target_sr.
    
    Args:
        waveform (Tensor): shape [1, T]
        orig_sr (int): original sampling rate
        target_sr (int): target sampling rate (default=8000)
    Returns:
        Tensor: resampled waveform
    """
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform


def load_and_preprocess_audio(audio_path=None, audio_byte=None, target_sr=8000, target_len=100):
    if audio_path is not None:
        tens, sr = torchaudio.load(audio_path)
    elif audio_byte is not None:
        with io.BytesIO(audio_byte) as f:
            tens, sr = torchaudio.load(f)
    else:
        raise ValueError("Provide either file path or audio bytes")
    
    tens = resample(tens, sr, target_sr)

    # A suggested quick fix for the improving robustness for microphone inputs
    # remove DC offset
    tens = tens - tens.mean()
    # RMS normalize to target RMS
    rms = tens.pow(2).mean().sqrt()
    target_rms = 0.05
    tens = tens * (target_rms / (rms + 1e-6))
    # optional: trim silence
    num_array = tens.numpy()
    num_array_trimmed, _ = librosa.effects.trim(num_array, top_db=20)
    tens = torch.from_numpy(num_array_trimmed)

    tens = preprocess(tens)
    mel_length = tens.shape[-1]
    if mel_length < target_len:
        tens = F.pad(tens, (0, target_len - mel_length))
    elif mel_length > target_len:
        tens = tens[..., :target_len]

    # ensure correct shape [batch, channel, freq, time]
    if tens.dim() == 2:
        tens = tens.unsqueeze(0)  # add batch
    if tens.size(1) != 1:
        tens = tens.unsqueeze(1)  # add channel if missing
    print(tens.shape)

    return tens
