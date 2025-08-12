import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader

class MLP:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
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
        data = np.array(data).reshape(1, -1)
        data = data.astype(np.float32)  # Ensure data is in float32 format
        answerConfidences = self.confidenceModel.predict(data)[0]
        answerVal = np.argmax(answerConfidences)
        return answerVal, answerConfidences[answerVal]