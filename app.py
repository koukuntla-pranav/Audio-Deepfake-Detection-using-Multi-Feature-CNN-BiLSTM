# app.py
import streamlit as st
import torch
import tempfile
import os
import sounddevice as sd
sd.query_devices()
from scipy.io.wavfile import write
from model import MultiFeatureCNN_BiLSTM
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

# Input Method: Upload or Record
option = st.radio("Choose input method", ["Upload", "Record with Mic"])

if option == "Upload":
    uploaded = st.file_uploader("Upload an audio file", type=["flac", "wav", "mp3", "ogg", "m4a"])
    if uploaded:
        file_suffix = os.path.splitext(uploaded.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name

        st.audio(temp_path)

elif option == "Record with Mic":
    duration = st.slider("🎙️ Recording duration (sec)", 1, 10, 5)
    if st.button("🔴 Start Recording"):
        st.info("Recording...")
        fs = 16000
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, fs, audio)
            temp_path = tmp.name

        st.audio(temp_path)

# Shared Prediction Logic
if "temp_path" in locals():
    try:
        mfcc, mel, cqt, cqcc = extract_features(temp_path)
        mfcc, mel, cqt, cqcc = mfcc.to(device), mel.to(device), cqt.to(device), cqcc.to(device)

        with torch.no_grad():
            output = model(mfcc, mel, cqcc, cqt).squeeze().item()
            pred = "✅ Bonafide" if output >= 0.5 else "❌ Spoof"
            st.success(f"**Prediction:** {pred}  \n**Score:** {output:.4f}")

    except Exception as e:
        st.error(f"Error processing the audio: {e}")

    os.remove(temp_path)
