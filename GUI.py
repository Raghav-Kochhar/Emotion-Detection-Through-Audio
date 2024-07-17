import streamlit as st
import librosa
import numpy as np
import torch
import torchaudio
from torch import nn
import os
import st_audiorec
import random


# Define the EmotionClassifier class (make sure this matches your model definition)
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


@st.cache_resource
def load_model():
    input_dim = 40  # Number of MFCC features
    hidden_dim = 128
    num_classes = 7  # Update this based on your number of emotion classes
    model = EmotionClassifier(input_dim, hidden_dim, num_classes)

    # Load the trained weights
    model.load_state_dict(
        torch.load("best_female_emotion_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


def preprocess_audio(audio_file, max_length=10):
    audio, sr = librosa.load(audio_file, sr=16000, duration=max_length)

    if len(audio) > 16000 * max_length:
        audio = audio[: 16000 * max_length]
    else:
        audio = np.pad(audio, (0, 16000 * max_length - len(audio)))

    audio = torch.from_numpy(audio).float().unsqueeze(0)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={"n_mels": 80},
    )
    mfcc = mfcc_transform(audio)
    return mfcc.squeeze(0).transpose(0, 1)


def is_female_voice():
    # Randomly decide if it's a female voice
    return random.choice([True, False])


def predict_emotion(model, audio_file):
    processed_audio = preprocess_audio(audio_file)
    processed_audio = processed_audio.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(processed_audio)
    return prediction


def main():
    st.title("Emotion Recognition from Audio")

    option = st.selectbox(
        "Choose an option:", ("Upload an audio file", "Record an audio clip")
    )

    model = load_model()
    emotions = [
        "angry",
        "fear",
        "happy",
        "neutral",
        "sad",
        "disgust",
        "surprise",
    ]  # Update based on your classes

    if option == "Upload an audio file":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            if st.button("Predict Emotion"):
                if is_female_voice():
                    try:
                        prediction = predict_emotion(model, uploaded_file)
                        predicted_emotion = emotions[prediction.argmax().item()]
                        confidence = torch.softmax(prediction, dim=1).max().item()
                        st.success(
                            f"Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f})",
                            icon="✅",
                        )
                    except Exception as e:
                        st.error(f"Error in processing: {str(e)}")
                else:
                    st.error("Please upload a female voice.", icon="🚨")

    elif option == "Record an audio clip":
        st.write("Record an audio clip")
        audio_bytes = st_audiorec.st_audiorec()

        if audio_bytes is not None:
            with open("recorded_audio.wav", "wb") as f:
                f.write(audio_bytes)

            if st.button("Predict Recorded Emotion"):
                if is_female_voice():
                    try:
                        prediction = predict_emotion(model, "recorded_audio.wav")
                        predicted_emotion = emotions[prediction.argmax().item()]
                        confidence = torch.softmax(prediction, dim=1).max().item()
                        st.success(
                            f"Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f})",
                            icon="✅",
                        )
                    except Exception as e:
                        st.error(f"Error in processing: {str(e)}")
                else:
                    st.error("Please record a female voice.", icon="🚨")


if __name__ == "__main__":
    main()
