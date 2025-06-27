# predict.py
import argparse
import torch
import librosa
import numpy as np
from model import MultiFeatureCNN_BiLSTM
from utils import normalize_and_pad
from utils import extract_features



def predict(file_path, model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiFeatureCNN_BiLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mfcc, mel, cqt, cqcc = extract_features(file_path)
    mfcc, mel, cqt, cqcc = mfcc.to(device), mel.to(device), cqt.to(device), cqcc.to(device)

    with torch.no_grad():
        output = model(mfcc, mel, cqcc, cqt).squeeze().item()
        label = "Bonafide ✅" if output >= 0.5 else "Spoof ❌"
        print(f"Prediction: {label} (Score: {output:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the input .flac file")
    parser.add_argument("--model", default="best_model.pth", help="Path to saved model")
    args = parser.parse_args()

    predict(args.file, args.model)
