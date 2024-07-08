import streamlit as st
import librosa
import numpy as np
import pandas as pd
from keras.models import model_from_json
import keras
import pickle
import st_audiorec


# Function to load model and perform prediction
@st.cache_resource
def load_model():
    # Load model architecture from JSON file
    json_file = open("model_json.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights("emotion_model.h5")

    # Compile model
    opt = keras.optimizers.Adam()
    loaded_model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    return loaded_model


# Function to preprocess audio file
def preprocess_audio(audio_file):
    X, sample_rate = librosa.load(
        audio_file, res_type="kaiser_fast", duration=2.5, sr=44100, offset=0.5
    )
    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)

    # Ensure the shape matches the model input shape
    if mfccs.shape[0] != 192:
        mfccs = np.pad(mfccs, (0, max(0, 192 - len(mfccs))), mode="constant")

    return pd.DataFrame(data=mfccs).T


# Function to check if the voice is female
def is_female_voice(audio_file):
    # Placeholder function: Implement actual female voice detection here
    # Returning True for simplicity
    return True


# Function to make predictions
def predict_emotion(model, audio_file):
    processed_audio = preprocess_audio(audio_file)
    processed_audio = np.expand_dims(processed_audio, axis=2)
    prediction = model.predict(processed_audio, batch_size=16, verbose=1)
    return prediction


# Main function
def main():
    st.title("Emotion Recognition from Audio")

    # Selectbox for choosing between Upload or Record
    option = st.selectbox(
        "Choose an option:", ("Upload an audio file", "Record an audio clip")
    )

    if option == "Upload an audio file":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            model = load_model()

            if st.button("Predict Emotion"):
                if is_female_voice(uploaded_file):
                    try:
                        prediction = predict_emotion(model, uploaded_file)

                        # Load label encoder
                        with open("labels", "rb") as f:
                            lb = pickle.load(f)

                        # Get predicted labels
                        final = prediction.argmax(axis=1)
                        final = final.astype(int).flatten()
                        final = lb.inverse_transform((final))
                        if final[0][:4] == "male":
                            st.error("Please record a female voice", icon="🚨")

                        else:
                            st.success(f"Predicted Emotion: {final[0]}", icon="✅")
                    except ValueError as e:
                        st.write(f"Error in processing: {e}")
                else:
                    st.write("Please upload a female voice.")

    elif option == "Record an audio clip":
        st.write("Record an audio clip")
        audio_bytes = st_audiorec.st_audiorec()

        if audio_bytes is not None:
            # Save recorded audio
            with open("recorded_audio.wav", "wb") as f:
                f.write(audio_bytes)

            model = load_model()

            if st.button("Predict Recorded Emotion"):
                if is_female_voice("recorded_audio.wav"):
                    try:
                        prediction = predict_emotion(model, "recorded_audio.wav")

                        # Load label encoder
                        with open("labels", "rb") as f:
                            lb = pickle.load(f)

                        # Get predicted labels
                        final = prediction.argmax(axis=1)
                        final = final.astype(int).flatten()
                        final = lb.inverse_transform((final))
                        if final[0][:4] == "male":
                            st.error("Please record a female voice", icon="🚨")
                        else:
                            st.success(f"Predicted Emotion: {final[0]}", icon="✅")
                    except ValueError as e:
                        st.write(f"Error in processing: {e}")
                else:
                    st.write("Please record a female voice.")


if __name__ == "__main__":
    main()
