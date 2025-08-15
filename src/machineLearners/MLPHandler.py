import machineLearners.MLP as mlp
import numpy as np
import tensorflow as tf

SAVE_MODEL_PATH = './data/models/mlp_model.keras'
REBUILD_MODEL = False

class MLPHandler:
    def __init__(self):
        self.mlp = mlp.MLP()

    def train(self, trainDataset, validationDataset):
        if REBUILD_MODEL:
            self.learn(trainDataset, validationDataset)
            return
        try:
            self.mlp.load_model(SAVE_MODEL_PATH)
        except Exception as e:  # Use Exception to catch all errors
            print("Error loading model:", e)
            self.learn(trainDataset, validationDataset)

    def learn(self, trainDataset, validationDataset):
        self.mlp.train(trainDataset, validationDataset)
        self.mlp.save_model(SAVE_MODEL_PATH)
        print("Model trained and saved.")

    def predict(self, data):
        data = np.array(data).reshape(1, -1)
        data = data.astype(np.float32)
        return self.mlp.predict(data)