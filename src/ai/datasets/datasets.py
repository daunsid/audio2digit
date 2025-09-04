
import torch
from torch.utils.data import Dataset


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
