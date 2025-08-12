import utils.MLPHelper as helper
import machineLearners.MLP as mlp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader

class MLPHandler:
    def __init__(self):
        self.mlp = mlp.MLP()

    def train(self, data, labels):
        try:
            self.mlp.load_model('./data/models/mlp_model.keras')
        except Exception as e:  # Use Exception to catch all errors
            print("Error loading model:", e)
            self.mlp.train(data, labels)
            self.mlp.save_model('./data/models/mlp_model.keras')
            print("Model trained and saved.")

    def predict(self, data):
       return self.mlp.predict(data)