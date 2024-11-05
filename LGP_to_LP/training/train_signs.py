# 1. Treinar LSTM
def LSTM():
    import os
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # print(gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    from tensorflow.keras.utils import to_categorical
    from prepare_data import data_ready_firstrain  # obter dados preparados para treino, teste, validação
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
    from tensorflow.keras.optimizers import Adam
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

    # obter dados de treino e teste
    sign, X_train, y_train, X_test, y_test = data_ready_firstrain()

    log_dir = os.path.join('graphs_lstm')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    # 1ª camada
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(90, 1662)))
    # 2ª camada
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    # 3ª camada
    model.add(LSTM(64, return_sequences=False, activation='relu'))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(sign.shape[0], activation='softmax'))

    # ajuste de parâmetros da rede
    path_model = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/all_models/lingua_gestual/'
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    es = EarlyStopping(monitor='categorical_accuracy', min_delta=1e-10, patience=100, verbose=1)
    rlr = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5, patience=25, verbose=1)
    mcp = ModelCheckpoint(filepath=path_model, monitor='categorical_accuracy', verbose=1, save_best_only=True,
                          save_weights_only=False)
    model.summary()

    # treinar rede
    model.fit(X_train, y_train, epochs=1000, callbacks=[es, mcp, rlr, tb_callback], validation_data=(X_test, y_test),
              batch_size=64)

    # previsão
    res = model.predict(X_test)
    print(sign[np.argmax(res[0])])
    print(sign[np.argmax(y_test[0])])

    '''Como o computador não tem capacidade suficiente para treinar mais do que 4950 exemplos é necessário retreinar
       o modelo.
       O modelo linua_gestual = 229 epochs -> accuracy = 0.91 / accuracy_validação = 0.8756
    '''


def new_LSTM():
    # https://github.com/BKaiwalya/Deep-Learning_Human-Activity-Recognition
    import os
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # print(gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from prepare_data import data_ready_firstrain

    # obter dados de treino e teste
    sign, X_train, y_train, X_test, y_test = data_ready_firstrain()

    log_dir = os.path.join('gráficos_tensorboard/graphs_LSTM')
    tb_callback = TensorBoard(log_dir=log_dir)

    inputs = tf.keras.Input(shape=(90, 1662))
    x = layers.LSTM(256, return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='LSTM_model')
    model.summary()

    path = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/all_models/lingua_gestual(LSTM)/'
    try:
        os.mkdir(path)
    except:
        pass

    # Compiling the new_lstm
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    es = EarlyStopping(monitor='categorical_accuracy', min_delta=1e-10, patience=40, verbose=1)
    rlr = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5, patience=25, verbose=1)
    mcp = ModelCheckpoint(filepath=path, monitor='categorical_accuracy', verbose=1, save_best_only=True,
                          save_weights_only=False)
    model.fit(X_train, y_train, epochs=10000, batch_size=32, callbacks=[es, mcp, rlr, tb_callback],
              validation_data=(X_test, y_test))

    # previsão
    res = model.predict(X_test)
    print(sign[np.argmax(res[0])])
    print(sign[np.argmax(y_test[0])])
    model.save('signs_LSTM', save_format='h5')


def GRU():
    import os
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # print(gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from prepare_data import data_ready_firstrain
    from tensorflow import keras

    # obter dados de treino e teste
    sign, X_train, y_train, X_test, y_test = data_ready_firstrain()

    log_dir = os.path.join('gráficos_tensorboard/graphs_GRU')
    tb_callback = TensorBoard(log_dir=log_dir)

    inputs = tf.keras.Input(shape=(90, 1662))
    x = layers.GRU(256, return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='GRU_model')
    model.summary()

    path = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/all_models/lingua_gestual(GRU)/'
    try:
        os.mkdir(path)
    except:
        pass

    # Compiling the new_lstm
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    es = EarlyStopping(monitor='categorical_accuracy', min_delta=1e-10, patience=40, verbose=1)
    rlr = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5, patience=25, verbose=1)
    mcp = ModelCheckpoint(filepath=path, monitor='categorical_accuracy', verbose=1, save_best_only=True,
                          save_weights_only=False)
    model.fit(X_train, y_train, epochs=10000, batch_size=32, callbacks=[es, rlr, mcp, tb_callback],
              validation_data=(X_test, y_test))

    # previsão
    res = model.predict(X_test)
    print(sign[np.argmax(res[0])])
    print(sign[np.argmax(y_test[0])])
    model.save('sings_gru', save_format='h5')


def RNN():
    import os
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # print(gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from prepare_data import data_ready_firstrain

    # obter dados de treino e teste
    sign, X_train, y_train, X_test, y_test = data_ready_firstrain()

    log_dir = os.path.join('gráficos_tensorboard/graphs_RNN')
    tb_callback = TensorBoard(log_dir=log_dir)

    inputs = tf.keras.Input(shape=(90, 1662))
    x = layers.SimpleRNN(256, return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.SimpleRNN(128, return_sequences=False)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='RNN_model')
    model.summary()

    path = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/all_models/lingua_gestual(RNN)/'
    try:
        os.mkdir(path)
    except:
        pass

    # Compiling the new_lstm
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    es = EarlyStopping(monitor='categorical_accuracy', min_delta=1e-10, patience=40, verbose=1)
    rlr = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5, patience=25, verbose=1)
    mcp = ModelCheckpoint(filepath=path, monitor='categorical_accuracy', verbose=1, save_best_only=True,
                          save_weights_only=False)
    model.fit(X_train, y_train, epochs=10000, batch_size=32, callbacks=[es, rlr, mcp, tb_callback],
              validation_data=(X_test, y_test))

    # previsão
    res = model.predict(X_test)
    print(sign[np.argmax(res[0])])
    print(sign[np.argmax(y_test[0])])
    model.save('sings_rnn', save_format='h5')


def BIdir():
    import os
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # print(gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from prepare_data import data_ready_firstrain

    # obter dados de treino e teste
    sign, X_train, y_train, X_test, y_test = data_ready_firstrain()

    log_dir = os.path.join('gráficos_tensorboard/graphs_alf')
    tb_callback = TensorBoard(log_dir=log_dir)

    inputs = tf.keras.Input(shape=(60, 1662))
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(26, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='BiLSTM_model')
    model.summary()

    path = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/all_models/alfa/'
    try:
        os.mkdir(path)
    except:
        pass

    # Compiling the new_lstm
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    es = EarlyStopping(monitor='categorical_accuracy', min_delta=1e-10, patience=40, verbose=1)
    rlr = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5, patience=25, verbose=1)
    mcp = ModelCheckpoint(filepath=path, monitor='categorical_accuracy', verbose=1, save_best_only=True,
                          save_weights_only=False)
    model.fit(X_train, y_train, epochs=10000, batch_size=32, callbacks=[es, rlr, mcp, tb_callback],
              validation_data=(X_test, y_test))

    # previsão
    res = model.predict(X_test)
    print(sign[np.argmax(res[20])])
    print(sign[np.argmax(y_test[20])])
    model.save('sings_alfa', save_format='h5')


if __name__ == '__main__':
    # LSTM()
    # new_LSTM()
    # GRU()
    # RNN()
    BIdir()
