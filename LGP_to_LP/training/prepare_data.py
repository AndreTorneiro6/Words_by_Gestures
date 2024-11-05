# 0. Preparação dos dados

from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def data_ready_firstrain():
    target = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/Dataset/LGP2LP/Keypoints/Declarativo/Holistic'
    signs = []
    # loop para obter nome de todas as pastas
    for root, dirs, files in os.walk(target):
        for f in files:
            words = root.split('/')
            if len(os.listdir(root)) > 2:
                if words[10] in signs:
                    # não coloca nome se já estiver na lista
                    pass
                else:
                    # adiciona nome do video à lista
                    signs.append(words[10])
    print((signs))

    # converter lista para um array
    sign = np.array(signs)

    # iteração para fazer labelling das classes
    labelling = {label: num for num, label in enumerate(sign)}
    # print(labelling)

    # # juntar todos os dados extraídos atraves do mediapipe
    sequences_train, sequences_test, labels_train, labels_test = [], [], [], []
    i = 0
    while i < len(signs):
        # obter quantidade de exemplos de cada gesto
        examples = os.listdir(os.path.join(target, signs[i]))
        print(os.path.join(target, signs[i]))
        k = 0
        while k < len(examples):
            video = []
            # print(os.path.join(target, signs[i], examples[k]))

            # obter número de frames de cada exemplo
            frame_size = os.listdir(os.path.join(target, signs[i], examples[k]))
            # print(os.path.join(target, signs[i], examples[k]))
            sequence_length = len(frame_size)
            # print(sequence_length)
            for frame_num in range(sequence_length):
                # Juntar conjunto de frames para formar sequência
                res = np.load(os.path.join(target, signs[i], examples[k], "{}.npy".format(frame_num)))
                # print(frame_num)
                video.append(res)

            if k < 100:
                sequences_test.append(video)
                labels_test.append(labelling[signs[i]])
            elif k < 400:
                # print(os.path.join(target, signs[i], examples[k]))
                sequences_train.append(video)
                # print(len(sequences_train))
                labels_train.append(labelling[signs[i]])
            else:
                break
            k += 1
        i += 1

    # apagar todas as variaveis para ter memória
    del video
    del signs
    del i
    del k
    del frame_num
    del frame_size
    del sequence_length
    del examples
    del res

    # dados treino
    X_train = np.array(sequences_train, dtype=np.float16)
    del sequences_train
    print(X_train.shape)
    y_train = to_categorical(labels_train).astype(int)
    del labels_train

    # dados teste
    X_test = np.array(sequences_test, dtype=np.float16)
    del sequences_test
    print(X_test.shape)
    y_test = to_categorical(labels_test).astype(int)
    del labels_test
    print('fim')
    del labelling

    return sign, X_train, y_train, X_test, y_test


def data_ready_retrain(test, train):
    target = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/Dataset/LGP2LP/Keypoints/Declarativo/Holistic'  # diretória
    signs = []
    # loop para obter nome de todas as pastas
    for root, dirs, files in os.walk(target):
        for f in files:
            words = root.split('/')
            if len(os.listdir(root)) > 2:
                if words[10] in signs:
                    # não coloca nome se já estiver na lista
                    pass
                else:
                    # adiciona nome do video à lista
                    signs.append(words[10])
    print(len(signs))

    # converter lista para um array
    sign = np.array(signs)

    # iteração para fazer labelling das classes
    labelling = {label: num for num, label in enumerate(sign)}
    # print(labelling)

    # # juntar todos os dados extraídos atraves do mediapipe
    sequences_train, sequences_test, labels_train, labels_test = [], [], [], []
    i = 0
    while i < len(signs):
        # obter quantidade de exemplos de cada gesto
        examples = os.listdir(os.path.join(target, signs[i]))
        print(os.path.join(target, signs[i]))
        k = 0
        while k < len(examples):
            video = []
            # print(os.path.join(target, signs[i], examples[k]))

            # obter número de frames de cada exemplo
            frame_size = os.listdir(os.path.join(target, signs[i], examples[k]))
            # print(os.path.join(target, signs[i], examples[k]))
            sequence_length = len(frame_size)
            # print(sequence_length)
            for frame_num in range(sequence_length):
                # Juntar conjunto de frames para formar sequência
                res = np.load(os.path.join(target, signs[i], examples[k], "{}.npy".format(frame_num)))
                # print(frame_num)
                video.append(res)

            if test-400 <= k < test:
                sequences_test.append(video)
                labels_test.append(labelling[signs[i]])
            elif train <= k < train+100:
                # print(os.path.join(target, signs[i], examples[k]))
                sequences_train.append(video)
                # print(len(sequences_train))
                labels_train.append(labelling[signs[i]])
            k += 1
        i += 1

    # apagar todas as variaveis para ter memória
    del video
    del signs
    del i
    del k
    del frame_num
    del frame_size
    del sequence_length
    del examples
    del res

    # dados treino
    X_train = np.array(sequences_train, dtype=np.float16)
    del sequences_train
    print(X_train.shape)
    y_train = to_categorical(labels_train).astype(int)
    del labels_train

    # dados teste
    X_test = np.array(sequences_test, dtype=np.float16)
    del sequences_test
    print(X_test.shape)
    y_test = to_categorical(labels_test).astype(int)
    del labels_test
    del labelling
    print('fim')

    return sign, X_train, y_train, X_test, y_test


def data_ready_lowmemory():
    target = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/Dataset/LGP2LP/Keypoints/Declarativo/Holistic'  # diretória
    signs = []
    # loop para obter nome de todas as pastas
    for root, dirs, files in os.walk(target):
        for f in files:
            words = root.split('/')
            if len(os.listdir(root)) > 2:
                if words[10] in signs:
                    # não coloca nome se já estiver na lista
                    pass
                else:
                    # adiciona nome do video à lista
                    signs.append(words[10])
    print(len(signs))


    # signs = ['Ele está gravemente ferido', 'O elevador está preso','Ele está inconsciente', 'Tenho uma erupção cutânea']
    # converter lista para um array
    sign = np.array(signs)

    # iteração para fazer labelling das classes
    labelling = {label: num for num, label in enumerate(sign)}
    # print(labelling)

    # # juntar todos os dados extraídos atraves do mediapipe
    sequences_train, sequences_test, labels_train, labels_test = [], [], [], []
    rounds = 0
    i = 0
    while i < len(signs):
        # obter quantidade de exemplos de cada gesto
        examples = os.listdir(os.path.join(target, signs[i]))
        print(os.path.join(target, signs[i]))
        if rounds == 0:
            k = 0
            x = 100
            y = 399
        elif rounds == 1:
            k = 400
            x = 400
            y = 699
        elif rounds == 2:
            k = 700
            x = 700
            y = 999
        print(k)
        while k < len(examples):
            video = []
            # print(os.path.join(target, signs[i], examples[k]))

            # obter número de frames de cada exemplo
            frame_size = os.listdir(os.path.join(target, signs[i], examples[k]))
            # print(os.path.join(target, signs[i], examples[k]))
            sequence_length = len(frame_size)
            # print(sequence_length)
            for frame_num in range(sequence_length):
                # Juntar conjunto de frames para formar sequência
                res = np.load(os.path.join(target, signs[i], examples[k], "{}.npy".format(frame_num)))
                # print(frame_num)
                video.append(res)

            if k < 100:
                sequences_test.append(video)
                labels_test.append(labelling[signs[i]])
                k += 1
            elif x <= k <= y:
                # print(os.path.join(target, signs[i], examples[k]))
                sequences_train.append(video)
                # print(len(sequences_train))
                # X = tf.ragged.constant(sequences_train)
                labels_train.append(labelling[signs[i]])
                # print(k)
                k += 1

            elif k > y:
                k = k
                break
        i += 1

        if i == 9:
            i = 0
            rounds += 1
        if rounds == 1 and i == 0:
            X1 = np.array(sequences_train, dtype=np.float16)
            sequences_train = []
            X_test = np.array(sequences_test)
            del sequences_test
        elif rounds == 2 and i == 0:
            X2 = np.array(sequences_train, dtype=np.float16)
            sequences_train = []
        elif rounds == 3:
            X3 = np.array(sequences_train, dtype=np.float16)
            break

        # print(i)
    del video
    del signs
    del sequences_train

    # dados treino
    y_train = to_categorical(labels_train).astype(int)
    del labels_train
    # print(X_test.shape)
    # print(X1.shape)
    # print(X2.shape)
    # print(X3.shape)
    X_train = np.concatenate((X1, X2, X3))
    print(X_train.shape)

    # dados teste
    y_test = to_categorical(labels_test).astype(int)
    del labels_test
    print('fim')

    return sign, X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # data_ready_lowmemory()
    # data_ready_retrain(900,650)
    data_ready_firstrain()
