# https://github.com/x4nth055/emotion-recognition-using-speech
import os
import time
import numpy as np

np.seterr(divide='ignore')
# np.seterr(divide = 'warn')

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io.wavfile import read
from scipy import signal
import random
import librosa.display
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def prepare_data_audio():
    # 0. Obtenção dos dados do dataset
    classes = []
    labels = []
    dataset = []
    sound_name = []

    target = glob.glob(r'C:\Users\andre\Desktop\Dataset\LP2LGP\audios/*')
    # print(target)

    # Iterate over class folders
    for index, dir in tqdm(enumerate(target)):
        class_name = os.path.basename(dir)
        classes.append(class_name)
        files = glob.glob(f'{dir}/*')
        for file in files:
            fs, audio = read(file)
            audio_chanel = audio[:, 0]
            dataset.append(audio_chanel)
            labels.append(index)
            sound_name.append(class_name)

    print('Dataset Adquirido')

    #  1. Aplicação de filtro  para frequencias vocais
    sample_rate = 44100

    # Frequência de Nyquist
    nyq_rate = sample_rate / 2.0

    # Frequência de corte do filtro
    cutoff_hz = [20 / nyq_rate, 3600 / nyq_rate]

    # Ordem do filtro
    N = 4

    b, a = signal.butter(N, cutoff_hz, 'bandpass')
    w, h = signal.freqz(b, a)

    graph0 = False  # mudar esta linha para visualizar gráfico

    if graph0:
        plt.rcParams['figure.figsize'] = [6, 2]
        plt.plot(sample_rate * w / (2 * np.pi), 20 * np.log10(abs(h)))
        plt.title('IIR Passa-Banda de 4º ordem de 20-3600Hz')
        plt.xlim([0, 4500])
        plt.ylim([-10, 5])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('dB')
        plt.show()

    # filtragem das amostras
    filtered_sounds = []
    for i in tqdm(range(len(dataset)), position=0, leave=True):
        filtered = signal.filtfilt(b, a, dataset[i])
        filtered_sounds.append(filtered)
    del dataset
    print('Amostras filtradas')

    #  2. Conversão dos dados para array

    X = np.asarray(filtered_sounds, dtype=np.float32)
    X = np.reshape(X, (X.shape[0], X.shape[1]))
    X = X / sample_rate
    y = np.array(labels)

    graph1 = False

    # gráficos com amostras
    if graph1:
        plt.clf()
        columns = 4
        rows = 4
        fig = plt.figure(figsize=(16, 8))
        for i in range(1, columns * rows + 1):
            r = random.randint(0, X.shape[0] - 1)
            fig.add_subplot(rows, columns, i)
            plt.plot(X[r])
            plt.title(str(labels[r]))
            plt.axis('off')
        plt.show()

    # 3. Observação das features das amostras
    amostras = True
    stft_visualizer = False
    tonnetz_visualizer = False
    mfccs_visualizer = True

    if amostras:
        m_class = ['Declarativo', 'Exclamativo', 'Interrogação']
        c = 0
        class_idx = []
        for i in range(len(sound_name)):
            if c < len(m_class):
                if m_class[c] in sound_name[i]:
                    class_idx.append(i)
                    c += 1

    # Análise do STFT (Espectograma)
    if stft_visualizer:
        plt.rcParams['figure.figsize'] = [15, 7]

        sp = class_idx
        for i in range(3):
            f, t, S1 = signal.spectrogram(X[sp[i]], sample_rate, window='flattop', nperseg=sample_rate // 10,
                                          noverlap=sample_rate // 20, scaling='spectrum', mode='magnitude')

            plt.subplot(2, 2, i + 1)
            plt.pcolormesh(t, f, S1[:][:])
            plt.title('STFT of ' + sound_name[sp[i]])
            plt.xlabel('time(s)')
            plt.ylabel('frequency(Hz)')
            plt.ylim(0, 1000)
        plt.show()

    # mfccs
    if mfccs_visualizer:
        plt.rcParams['figure.figsize'] = [19, 10]
        a = 0
        for i in range(12):
            mfccs = librosa.feature.mfcc(X[i + a], sample_rate, n_mfcc=128)
            plt.subplot(4, 3, i + 1)
            a = a + 400
            plt.title(sound_name[i + a])
            librosa.display.specshow(mfccs, x_axis='time')
            plt.xlim(0, 4)
            plt.tight_layout(pad=0.5)
            if i == 2:
                a = 1
            elif i == 5:
                a = 2
            elif i == 8:
                a = 3
        plt.show()

    # tonnetz
    if tonnetz_visualizer:
        plt.rcParams['figure.figsize'] = [19, 10]

        sp = class_idx
        a = 0
        for i in range(12):
            mel = librosa.feature.tonnetz(y=librosa.effects.harmonic(X[i + a]), sr=sample_rate)
            plt.subplot(4, 3, i + 1)
            a = a + 400
            plt.title(sound_name[i + a])
            img = librosa.display.specshow(mel, x_axis='time')
            plt.xlim(0, 4)
            plt.tight_layout(pad=0.5)
            if i == 2:
                a = 1
            elif i == 5:
                a = 2
            elif i == 8:
                a = 3
        plt.show()

    # 4. Extração das features
    all_features = []
    for i in tqdm(range(len(X))):
        mfccs = librosa.feature.mfcc(y=X[i], sr=sample_rate, n_mfcc=128)

        # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X[i]), sr=sample_rate)

        features = np.concatenate((mfccs, tonnetz))
        all_features.append(features)
    print('Features retiradas')

    X_features = np.array(all_features)
    print(X_features.shape)
    X_features = np.reshape(X_features, (X_features.shape[0], X_features.shape[1], X_features.shape[2], 1))
    print("Image shape: ", X_features.shape)
    print("\nImage Max: ", X_features.max().round(1), "\nImage Min: ", X_features.min().round(1))
    print("Image Mean: ", X_features.mean().round(1), "\nImage std: ", X_features.std().round(1))

    # 5.Normalização dos dados
    Xmean = X_features.mean().round(1)
    Xstd = X_features.std().round(1)
    X_features = (X_features - Xmean) / Xstd
    print(X_features.mean(), X_features.std())

    lb = LabelEncoder()
    y_features = to_categorical(lb.fit_transform(y))
    num_classes = len(lb.classes_)

    return X_features, y_features, num_classes


if __name__ == '__main__':
    prepare_data_audio()
