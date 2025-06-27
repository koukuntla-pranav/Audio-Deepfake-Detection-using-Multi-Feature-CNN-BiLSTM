# app.py
import streamlit as st
import torch
import librosa
import numpy as np
from model import MultiFeatureCNN_BiLSTM
from utils import normalize_and_pad
from utils import extract_features

st.set_page_config(page_title="Audio Deepfake Detector", layout="centered")

st.title("🎤 Audio Deepfake Detection (CNN-BiLSTM)")

@st.cache_resource
def load_model(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiFeatureCNN_BiLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

uploaded = st.file_uploader("Upload a .flac or .wav file", type=["flac", "wav"])
if uploaded:
    with open("temp_input.flac", "wb") as f:
        f.write(uploaded.read())
    
    st.audio("temp_input.flac", format="audio/flac")
    
    mfcc, mel, cqt, cqcc = extract_features("temp_input.flac")
    mfcc, mel, cqt, cqcc = mfcc.to(device), mel.to(device), cqt.to(device), cqcc.to(device)

    with torch.no_grad():
        output = model(mfcc, mel, cqcc, cqt).squeeze().item()
        pred = "✅ Bonafide" if output >= 0.5 else "❌ Spoof"
        st.success(f"**Prediction:** {pred}  \n**Score:** {output:.4f}")

