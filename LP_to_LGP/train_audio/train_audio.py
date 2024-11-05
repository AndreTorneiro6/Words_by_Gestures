import os
import numpy as np
from features_extraction import prepare_data_audio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


X_features, y_features, num_classes = prepare_data_audio()

X_train, X_test, y_train, y_test = train_test_split(X_features, y_features, test_size=0.2, random_state=1, stratify=y_features)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)
print(X_train.shape, X_val.shape, X_test.shape)

model = Sequential([
    Input(shape=(X_features.shape[1], X_features.shape[2], 1)),
    Conv2D(32, 3, activation='relu', padding="same"),
    MaxPooling2D(2, 2, padding='same'),
    Dropout(0.1),
    Conv2D(64, 3, activation='relu', padding="same"),
    MaxPooling2D(2, 2, padding='same'),
    Dropout(0.1),
    Conv2D(64, 3, activation='relu', padding="same"),
    MaxPooling2D(2, 2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(24, activation='relu'),
    Dense(num_classes, activation='softmax'),
])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["categorical_accuracy"])
# log_dir = os.path.join('graphs_audio')
# tb_callback = TensorBoard(log_dir=log_dir)

# path = r'C:\Users\andre\Desktop\WBG-words_by_gestures\all_models\audio'
# if not os.path.exists(path):
#     os.mkdir(path)

es = EarlyStopping(monitor='val_categorical_accuracy', patience=20, restore_best_weights=True)
# mcp = ModelCheckpoint(filepath=path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
#                       save_weights_only=False)

model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), callbacks=[es])

y_pred = model.predict(X_test)
print("Accuracy of test dataset: ", np.round(accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)), 4))

# # model.save('audio', save_format='h5')
import matplotlib.pyplot  as plt
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# from features_extraction import prepare_data_audio
# import numpy as np
import pandas as pd
#
# y_pred = model.predict(X_test)
labels = ['0', '1', '2']
plt.rcParams['figure.figsize'] = [5, 5]
print('Confusion Matrix:')
cm = confusion_matrix(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
cmd = ConfusionMatrixDisplay(cm, display_labels = labels)
cmd.plot()
print('Classification Report')
print(classification_report(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)))
plt.show()