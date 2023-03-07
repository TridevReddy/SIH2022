import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools
from google.colab import drive
drive.mount('/content/drive')

INPUT_DIR = '/content/drive/MyDrive/Human_audios_2/FE/'
OUTPUT_DIR = '/content/drive/MyDrive/'


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images-new')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images-new'))

i=0
for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
      if i<=1049:
        file_path = os.path.join(INPUT_DIR, filename)
        file_stem = Path(file_path).stem
        target_dir = f'class_{file_stem[0]}'
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images-new'), target_dir)
        file_dist_path = os.path.join(dist_dir, file_stem)
        if not os.path.exists(file_dist_path + '.png'):
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)
            file_stem = Path(file_path).stem
            sound_info, frame_rate = get_wav_info(file_path)
            pylab.specgram(sound_info, Fs=frame_rate)
            pylab.savefig(f'{file_dist_path}.png')
            pylab.close()
            i+=1
            
import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
%matplotlib inline


def create_spectrogram(audio_file, image_file):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    fig.savefig(image_file)
    plt.close(fig)

def create_pngs_from_wavs(input_path, output_path):

    dir = os.listdir(input_path)
    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        file_name = os.path.basename(file)
        file_name = file_name.replace("_", "")
        output_file = os.path.join(output_path, file_name.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)
        
from keras.preprocessing import image
import tensorflow as tf
#import tf.keras.utils.load_img
from tensorflow.keras.utils import img_to_array
def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(img_to_array(tf.keras.utils.load_img(os.path.join(path, file), target_size=(256, 256, 3))))
        labels.append((label))
    return images, labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)
        
x = []
y = []


import os

images, labels = load_images_from_path('/content/drive/MyDrive/images/human/', 1)
show_images(images)
    
x += images
y += labels

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

nsamples, nx, ny, nrgb = x_train_norm.shape
x_train2 = x_train_norm.reshape((nsamples,nx*ny*nrgb))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train2, y_train_encoded)

from sklearn.metrics import accuracy_score
nsamples, nx, ny, nrgb = x_test_norm.shape
x_test2 = x_test_norm.reshape((nsamples,nx*ny*nrgb))
y_pred = model.predict(x_test2)
print(y_pred)
print(accuracy_score(y_test_encoded, y_pred=y_pred))


#TESTING

# Human/Robo Detection

import librosa
import matplotlib.pyplot as plt
import keras.utils as image
import pickle
import numpy as np
import librosa.display

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

y, sr = librosa.load("/content/dr-help-me-this-dubstep-is-crazy-robo-pre-drop.wav")
ms = librosa.feature.melspectrogram(y, sr=sr)
log_ms = librosa.power_to_db(ms, ref=np.max)
librosa.display.specshow(log_ms, sr=sr)
fig.savefig("/content/1e.png")
plt.close(fig)

images_new = []
new_img = images_new.append(image.img_to_array(image.load_img("1e.png", target_size=(256, 256, 3))))
x_new = []
x_new += images_new

x_new_norm = np.array(x_new) / 255
nsamples, nx, ny, nrgb = x_new_norm.shape
x_new2 = x_new_norm.reshape((nsamples,nx*ny*nrgb))

loaded_model = pickle.load(open("rf_img.sav", 'rb'))

#a = model.predict(x_new2)
#print(a[0][0])

print("Robotic Call: {}".format(loaded_model.predict_proba(x_new2)[0][0]*100))
print("Human Call: {}".format(loaded_model.predict_proba(x_new2)[0][1]*100))
