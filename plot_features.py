import argparse
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(y, sr, out_dir):
    plt.figure(figsize=(10, 3))
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
    plt.title("Raw Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "waveform.png"))
    plt.close()

def plot_feature_matrix(feature, title, out_file):
    plt.figure(figsize=(10, 4))
    plt.imshow(feature, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Feature Index")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def process_and_plot(file_path, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    y, sr = librosa.load(file_path, sr=16000)
    print(f"Loaded: {file_path} | Duration: {len(y)/sr:.2f}s")

    # Plot raw waveform
    plot_waveform(y, sr, out_dir)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    plot_feature_matrix(mfcc, "MFCC (20 coefficients)", os.path.join(out_dir, "mfcc.png"))

    # Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plot_feature_matrix(mel_db, "Mel-Spectrogram (dB)", os.path.join(out_dir, "mel.png"))

    # CQT
    cqt = librosa.cqt(y=y, sr=sr, n_bins=20)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    plot_feature_matrix(cqt_db, "CQT (Constant-Q Transform)", os.path.join(out_dir, "cqt.png"))

    print(f"✅ Plots saved in '{out_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .flac audio file")
    parser.add_argument("--out", type=str, default="plots", help="Directory to save output images")
    args = parser.parse_args()

    process_and_plot(args.file, args.out)
