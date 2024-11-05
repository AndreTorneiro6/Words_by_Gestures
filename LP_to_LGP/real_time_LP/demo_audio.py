import os
import numpy as np
import wave
from tqdm import tqdm
from scipy.io.wavfile import read
from scipy import signal
import librosa.display
import glob
from tensorflow import keras
import speech_recognition as sr
import pyaudio
np.seterr(divide='ignore')
# np.seterr(divide = 'warn')


def record_audio():

    # variaveis do microfone
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5

    # nome do ficheiro a analisar
    file = 'real.wav'

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("----------------- recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # identify_intonation(file)


def identify_intonation(file):

    fs, audio = read(file)

    nyq_rate = fs / 2.0

    # Frequência de corte do filtro
    cutoff_hz = [20 / nyq_rate, 3600 / nyq_rate]

    # Ordem do filtro
    N = 4

    # filtragem do sinal
    b, a = signal.butter(N, cutoff_hz, 'bandpass')
    filtered = signal.filtfilt(b, a, audio[:, 0])

    # conversão para array
    X = np.expand_dims(np.array(filtered, dtype=np.float32), axis=0)
    X = np.reshape(X, (X.shape[0], X.shape[1]))
    X = X / fs

    # extrair features
    all_features = []
    for i in tqdm(range(len(X))):
        mfccs = librosa.feature.mfcc(y=X[i], sr=fs, n_mfcc=128)
        all_features.append(mfccs)

    X_features = np.array(all_features)
    X_features = np.reshape(X_features, (X_features.shape[0], X_features.shape[1], X_features.shape[2], 1))
    print("Image shape: ", X_features.shape)
    print("\nImage Max: ", X_features.max().round(1), "\nImage Min: ", X_features.min().round(1))
    print("Image Mean: ", X_features.mean().round(1), "\nImage std: ", X_features.std().round(1))

    # 5. Pré-pocessamento e treino
    Xmean = X_features.mean().round(1)
    Xstd = X_features.std().round(1)
    X_features = (X_features - Xmean) / Xstd
    print(X_features.mean(), X_features.std())

    model = keras.models.load_model(r'C:\Users\andre\Desktop\WBG-words_by_gestures\all_models\audio')
    model.summary()

    y_pred1 = model.predict(X_features)[0]
    print(y_pred1)

    type = np.array(['Declarativo', 'Exclamativo', 'Interrogação'])
    print(type[np.argmax(y_pred1)])
    # speech_2_text(file)


def speech_2_text():
    r = sr.Recognizer()
    harvard = sr.AudioFile(r'C:\Users\andre\Desktop\WBG-words_by_gestures\LP_to_LGP\real_time_LP\real.wav')
    with harvard as source:
        audio = r.record(source)
    print(r.recognize_google(audio, show_all=True, language='pt-PT'))


if __name__ == '__main__':
    # from tensorflow import keras
    # from keras.utils.vis_utils import plot_model
    # model = keras.models.load_model(r'C:\Users\andre\Desktop\WBG-words_by_gestures\all_models\audio')
    # model.summary()
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # record_audio()
    # identify_intonation()
    speech_2_text()
