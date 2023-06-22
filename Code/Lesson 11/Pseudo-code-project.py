# 1). Gather a dataset of audio recordings paired with their corresponding transcriptions (text labels).
# 2). Text Vectorization: Preprocess the transcriptions by removing punctuation, converting text to lowercase, and handling any other necessary text normalization steps.
# 3). Feature Extraction and Combination
# 4). Model Training
# 5). Model Evaluation
# 6). Model Optimization and Fine-tuning
# 7). Model Deployment


# Data repository : http://www.openslr.org/12
# https://github.com/besacier/ALFFA_PUBLIC/tree/master/ASR/SWAHILI/data/train/wav/SWH-05-20101106

import librosa # pip install librosa - for audio processing, you may need to use additional libraries like librosa or python_speech_features ,audio feature extraction or audio-to-text conversion
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Step 1: Load and process audio data
def extract_audio_features(audio_path):
    # Load audio file
    audio, sr = librosa.load(audio_path)    
    # Extract audio features (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)    
    # Flatten and normalize the feature matrix
    normalized_mfccs = np.mean(mfccs, axis=1)    
    return normalized_mfccs 

# Example usage:
audio_path = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Audio-Dataset\\SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part10.wav"
audio_features = extract_audio_features(audio_path)

# Step 2: Prepare the data
# Assuming you have a list of audio paths and corresponding transcriptions
audio1 =  os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Audio-Dataset\\SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part10.wav"
audio2 =  os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Audio-Dataset\\SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part100.wav"  

audio_paths = [audio1,audio2]
transcriptions = ['rais wa tanzania jakaya mrisho kikwete', 'rais wa tanzania jakaya mrisho kikwete']

# Extract audio features for all audio samples
audio_features_list = [extract_audio_features(path) for path in audio_paths]

# Convert transcriptions to numerical labels
label_mapping = {label: idx for idx, label in enumerate(set(transcriptions))}
numerical_labels = np.array([label_mapping[label] for label in transcriptions])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_features_list, numerical_labels, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)















