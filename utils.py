import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset


def load_protocol(protocol_path):
    data = []
    with open(protocol_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            filename = parts[1]  # Audio filename
            label = 1 if parts[-1] == "bonafide" else 0  # Convert to binary label (1=real, 0=fake)
            data.append((filename, label))
    return data


def normalize_and_pad(x, max_len=100):
    if x.shape[1] < max_len:
        pad_width = max_len - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
    else:
        x = x[:, :max_len]
    return x

def extract_features(file_path, max_len=100):
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    cqt = librosa.cqt(y=y, sr=sr, n_bins=20)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    cqcc = np.zeros_like(mfcc)  # Placeholder (if no CQCC extractor)

    return (
    torch.tensor(normalize_and_pad(mfcc, max_len), dtype=torch.float32).unsqueeze(0).unsqueeze(1),
    torch.tensor(normalize_and_pad(mel_db, max_len), dtype=torch.float32).unsqueeze(0).unsqueeze(1),
    torch.tensor(normalize_and_pad(cqt_db, max_len), dtype=torch.float32).unsqueeze(0).unsqueeze(1),
    torch.tensor(normalize_and_pad(cqcc, max_len), dtype=torch.float32).unsqueeze(0).unsqueeze(1),
)

class SpoofDataset(Dataset):
    def __init__(self, dataset_path, protocol_path, max_len=100):
        self.dataset_path = dataset_path
        self.data = load_protocol(protocol_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, label = self.data[idx]
        file_path = os.path.join(self.dataset_path, filename + ".flac")

        y, sr = librosa.load(file_path, sr=16000)
        y, _ = librosa.effects.trim(y)

        def normalize_and_pad(x):
            if x.shape[1] < self.max_len:
                pad_width = self.max_len - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
            else:
                x = x[:, :self.max_len]
            return x

        mfcc = normalize_and_pad(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20))
        mel = normalize_and_pad(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20), ref=np.max))
        cqt = normalize_and_pad(librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, n_bins=20)), ref=np.max))
        cqcc = normalize_and_pad(np.zeros_like(mfcc))  # Placeholder

        # Add channel dimension: (1, 20, 100)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        cqt = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0)
        cqcc = torch.tensor(cqcc, dtype=torch.float32).unsqueeze(0)

        return (mfcc, mel, cqt, cqcc), torch.tensor(label, dtype=torch.float32)
