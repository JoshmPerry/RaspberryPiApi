import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader

class CNN:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16,16,1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.confidenceModel = None

    def train(self, data, labels):
        # Scramble (shuffle) data order before splitting
        idx = np.random.permutation(len(data))
        x_all = np.array(data)[idx]
        y_all = np.array(labels)[idx]

        x_all = x_all.astype(np.float32)
        if len(x_all.shape) == 3:
            # Add channel dimension if missing (e.g., grayscale images)
            x_all = np.expand_dims(x_all, -1)

        # Split into train/test (e.g., 80/20 split)
        split_idx = int(0.8 * len(x_all))
        x_train, x_test = x_all[:split_idx], x_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]
        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=10)
        self.model.evaluate(x_test, y_test)

        self.confidenceModel = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
    
    def save_model(self, path):
        if self.confidenceModel is None:
            raise ValueError("Model has not been trained yet.")
        self.confidenceModel.save(path)
    
    def load_model(self, path):
        self.confidenceModel = tf.keras.models.load_model(path)

    def predict(self, data):
        if self.confidenceModel is None:
            raise ValueError("Model has not been trained yet.")
        data = np.array(data)
        if len(data.shape) == 3:
            data = np.expand_dims(data, -1)
        data = data.reshape(1, *data.shape[-3:])  # Ensure batch dimension
        data = data.astype(np.float32)
        answerConfidences = self.confidenceModel.predict(data)[0]
        answerVal = np.argmax(answerConfidences)
        return answerVal, answerConfidences[answerVal]