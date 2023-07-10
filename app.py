import os
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
import wave
import io
import librosa

app = Flask(__name__)

# Load the trained model
model_path = 'ChicDeviceModel.h5'
model = load_model(model_path)

# Define the target labels
labels = ['normal', 'mild', 'extreme']

# Audio configuration
CHANNELS = 1
RATE = 44100

def extract_features(audio_data, max_length=36):
    # Convert the audio data to a numpy array
    audio_data = np.array(audio_data)

    # Scale the audio data to the range [-1, 1]
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Save the audio data as a temporary WAV file
    temp_path = 'temp.wav'
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())

    # Read the temporary WAV file using librosa
    audio, _ = librosa.load(temp_path, sr=RATE)

    # Remove the temporary WAV file
    os.remove(temp_path)

    # Extract the Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=RATE, n_mfcc=20)

    # Pad or truncate the MFCCs to the fixed length
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]

    # Reshape the MFCCs into a 4D array
    mfccs = mfccs[np.newaxis, ..., np.newaxis]

    # Return the reshaped MFCCs as the feature
    return mfccs

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for audio classification
@app.route('/classify', methods=['POST'])
def classify():
    # Get the audio data from the request
    audio_data = request.get_json()['audio_data']

    # Extract features from the audio data
    features = extract_features(audio_data)

    # Perform the prediction
    prediction = model.predict(features)
    predicted_label = labels[np.argmax(prediction)]

    # Return the predicted label
    return predicted_label


if __name__ == '__main__':
    app.run(debug=True)
