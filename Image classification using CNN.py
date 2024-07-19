# Importing Necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

# Importing the MNIST Dataset
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Shape of X_train : ", X_train.shape)
print("Shape of X_test : ", X_test.shape)
print("Shape of y_train : ", y_train.shape)
print("Shape of y_test : ", y_test.shape)

# Loading Sample Images
plt.figure(figsize=(20, 20))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i])
plt.show()

plt.figure(figsize=(20, 20))
sample_indices = [150, 162, 178, 193, 205, 3978, 456, 7896, 57, 31897]
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[sample_indices[i]])
plt.show()

# Data Preprocessing
print("Shape of X_train : ", X_train.shape)
print("Shape of X_test : ", X_test.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
print("Shape of X_train : ", X_train.shape)
print("Shape of X_test : ", X_test.shape)

# One-Hot Encoding
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)
print(y_cat_train[0:11])

# Scaling feature data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Model Creation
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Model Training
model.fit(X_train, y_cat_train, epochs=50, callbacks=[early_stop], validation_data=(X_test, y_cat_test))
print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the model as mnist.h5")

# Model Performance
training_metrics = pd.DataFrame(model.history.history)
training_metrics.head()
training_metrics[['loss', 'val_loss']].plot()
training_metrics[['accuracy', 'val_accuracy']].plot()
score = model.evaluate(X_test, y_cat_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Model Predictions
predictions = np.argmax(model.predict(X_test), axis=-1)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Predicting Individual Images
new_img = X_test[95]
plt.imshow(new_img)
print("True label:", y_test[95])
print("Predicted label:", np.argmax(model.predict(new_img.reshape(1, 28, 28, 1)), axis=-1))

new_img2 = X_test[0]
plt.imshow(new_img2)
print("True label:", y_test[0])
print("Predicted label:", np.argmax(model.predict(new_img2.reshape(1, 28, 28, 1)), axis=-1))

new_img3 = X_test[397]
plt.imshow(new_img3)
print("True label:", y_test[397])
print("Predicted label:", np.argmax(model.predict(new_img3.reshape(1, 28, 28, 1)), axis=-1))
