# ==========================================================
# Deepfake Audio Detection Web App (Phase 1: Audio Only)
# Author: Your Name
# ==========================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
import tempfile
import os

# ---------------------------------------
# 1Page Configuration
# ---------------------------------------
st.set_page_config(page_title="Deepfake Audio Detector", layout="centered")
st.title(" Deepfake Audio Detection (Audio-only Phase)")
st.markdown("Upload a **video (.mp4)** or **audio (.wav)** file — "
            "the system will extract the audio, compute MFCC features, "
            "and use your trained CNN + LSTM model to classify it as **Real or Fake.**")

# ---------------------------------------
# 2️Load Trained Model
# ---------------------------------------
MODEL_PATH = "cnn_lstm_v2_best_model.h5"

@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

try:
    model = load_trained_model()
    st.success(" Model loaded successfully!")
except Exception as e:
    st.error(f" Could not load model: {e}")
    st.stop()

# ---------------------------------------
# 3️File Upload
# ---------------------------------------
uploaded_file = st.file_uploader(" Upload Audio or Video", type=["wav", "mp4", "avi"])

# Default decision threshold
THRESHOLD = 0.49

if uploaded_file is not None:
    st.info("Processing your file... please wait ")

    # Save uploaded file temporarily
    temp_path = tempfile.NamedTemporaryFile(delete=False).name
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # ---------------------------------------
    # 4Extract Audio (if Video)
    # ---------------------------------------
    if file_ext in [".mp4", ".avi"]:
        st.write(" Extracting audio from video...")
        try:
            clip = VideoFileClip(temp_path)
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            clip.audio.write_audiofile(temp_audio.name, codec='pcm_s16le', verbose=False, logger=None)
            audio_path = temp_audio.name
        except Exception as e:
            st.error(f"Error extracting audio: {e}")
            st.stop()
    else:
        audio_path = temp_path

    # ---------------------------------------
    # 5Load Audio & Compute MFCC
    # ---------------------------------------
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        st.audio(audio_path, format="audio/wav")
        st.write(f"🕒 Duration: {duration:.2f}s | Sample rate: {sr} Hz")

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < 300:
            mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :300]

        X_input = np.expand_dims(mfcc.T, axis=0)
        st.write("MFCC shape:", X_input.shape)

        # ---------------------------------------
        # 6Predict
        # ---------------------------------------
        prob = float(model.predict(X_input, verbose=0)[0][0])
        label = " REAL" if prob < THRESHOLD else " FAKE"

        st.subheader(" Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**Model Probability:** {prob:.4f}")
        st.write(f"**Threshold:** {THRESHOLD:.2f}")

        # ---------------------------------------
        # 7Visualization
        # ---------------------------------------
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax)
        ax.set_title(f"MFCC Spectrogram – {label} ({prob:.3f})")
        plt.colorbar(format="%+2.f dB")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("⬆️ Upload a .wav or .mp4 file to start analysis.")

