## Emotion Recognition from Audio

This Streamlit application enables emotion prediction from audio files, specifically designed for female voices. Users can either upload existing WAV files or record new audio clips directly through the interface. The application preprocesses the audio using MFCC features and applies a pre-trained deep learning model to predict emotions such as happiness, sadness, surprise, and more.

### Dataset:
- Surrey Audio-Visual Expressed Emotion [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
- Ryerson Audio-Visual Database of Emotional Speech and Song [(RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- Toronto emotional speech set [(TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- Crowd-sourced Emotional Mutimodal Actors Dataset [(CREMA-D)](https://www.kaggle.com/datasets/ejlok1/cremad)

### Features:
- **Upload Audio:** Upload WAV files for emotion prediction.
- **Record Audio:** Record audio clips using the integrated recorder.
- **Emotion Prediction:** Utilizes a deep learning model trained on emotional speech datasets to predict the predominant emotion in the audio clip.
- **Female Voice Detection:** Warns if the uploaded or recorded audio is not from a female voice.

### Technologies Used:
- Python, Streamlit
- Keras (TensorFlow backend)
- Librosa for audio processing

### Getting Started:
1. Clone the repository.
2. Install dependencies (`requirements.txt`).
3. Run `streamlit run app.py` to launch the application locally.

### Note:
- Ensure a microphone is connected for recording functionality.
