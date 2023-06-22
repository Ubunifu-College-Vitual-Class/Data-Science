import librosa # pip install librosa - for audio processing, you may need to use additional libraries like librosa or python_speech_features ,audio feature extraction or audio-to-text conversion
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


my_transcription_file = "E:\\Developers\\Audio-Dataset\\ALFFA_PUBLIC\\ASR\\SWAHILI\\data\\train\\text"
my_path_to_folder_with_wav = "E:\\Developers\\Audio-Dataset\\ALFFA_PUBLIC\\ASR\\SWAHILI\\data\\train\\wav"

# Using readlines()
transacription_file = open(my_transcription_file, 'r')
Lines = transacription_file.readlines(20000)

transcriptions = [] # list to hold the transcription
audio_paths = [] # list to hold it corresponding audio path

# Strips the newline character
for line in Lines:
    line_parts_array = line.split(None,1) #split along the 1st  whitespace
    first_part_of_text_string = line_parts_array[0]    
    name_of_first_folder = first_part_of_text_string.split('_',1) 
    full_folder_path = f'{my_path_to_folder_with_wav}\{name_of_first_folder[0]}\{first_part_of_text_string}.wav' # concatenate full path
    transcriptions.append(line_parts_array[1])
    audio_paths.append(full_folder_path)


# Data Preprocessing and Feature Extraction
features = []
max_frames = 0
for audio_file in audio_paths:
    # Load audio file
    audio, sr = librosa.load(audio_file)    
    # Extract audio features (MFCCs)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr )    
    # Append MFCCs to features list
    features.append(np.mean(mfcc, axis=1))    


# Convert transcriptions to numerical labels
label_mapping = {label: idx for idx, label in enumerate(set(transcriptions))}
y_labels = np.array([label_mapping[label] for label in transcriptions])


# Flatten and Prepare Data for Training
X = np.vstack(features)
y = np.array(y_labels)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Algorithm Selection and Training
# In this example, we'll use logistic regression as the supervised learning algorithm
model = LogisticRegression(max_iter=10000) # Increase max_iter value to 10000
model.fit(X_train, y_train)


# Evaluation (Optional)
accuracy = model.score(X_test, y_test)
print("Logistic regression accuracy:", accuracy)


# Train the model using Random forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# Evaluate Random forest model
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)


y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)


print("Random forest train accuracy:", train_accuracy)
print("Random forest test accuracy:", test_accuracy)























