import utils.MLPHelper as helper
import machineLearners.CNN as cnn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader

SAVE_MODEL_PATH = './data/models/cnn_model.keras'
REBUILD_MODEL = False

class CNNHandler:
    def __init__(self):
        self.cnn = cnn.CNN()

    def train(self, data, labels):
        # change data from (N, 256) to (N, 16, 16, 1)
        reformattedData = data.reshape(data.shape[0], 16, 16, 1)
        if REBUILD_MODEL:
            self.cnn.train(reformattedData, labels)
            self.cnn.save_model(SAVE_MODEL_PATH)
            print("Model trained and saved.")
            return
        try:
            self.cnn.load_model(SAVE_MODEL_PATH)
        except Exception as e:  # Use Exception to catch all errors
            print("Error loading model:", e)
            self.cnn.train(reformattedData, labels)
            self.cnn.save_model(SAVE_MODEL_PATH)
            print("Model trained and saved.")

    def predict(self, data):
       data = np.array(data)
       reformattedData = data.reshape(1, 16, 16, 1)
       return self.cnn.predict(reformattedData)