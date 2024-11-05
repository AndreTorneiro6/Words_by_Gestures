from tensorflow import keras
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from features_extraction import prepare_data_audio
import numpy as np
import pandas as pd

X_features, y_features, num_classes = prepare_data_audio()
X_train, X_test, y_train, y_test = train_test_split(X_features, y_features, test_size=0.2, random_state=1)

print(X_test.shape)


model = keras.models.load_model(r'C:\Users\andre\Desktop\WBG-words_by_gestures\all_models\audio')
model.summary()
y_pred = model.predict(X_test)
labels = ['0', '1', '2']
plt.rcParams['figure.figsize'] = [5, 5]
print('Confusion Matrix:')
cm = confusion_matrix(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
cmd = ConfusionMatrixDisplay(cm, display_labels = labels)
cmd.plot()
print('Classification Report')
print(classification_report(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)))
plt.show()

