import machineLearners.vit.VIT as vit
import numpy as np
import tensorflow as tf

SAVE_MODEL_PATH = './data/models/vit_model.keras'
REBUILD_MODEL = False

class VITHandler:
    def __init__(self):
        self.vit = vit.VIT(14)

    def train(self, trainDataset, validationDataset):
        if REBUILD_MODEL:
            self.learn(trainDataset, validationDataset)
            return
        try:
            self.vit.load_model(SAVE_MODEL_PATH)
        except Exception as e:  # Use Exception to catch all errors
            print("Error loading model:", e)
            self.learn(trainDataset, validationDataset)

    def learn(self, trainDataset, validationDataset):
        self.vit.train(trainDataset, validationDataset)
        self.vit.save_model(SAVE_MODEL_PATH)
        print("Model trained and saved.")

    def predict(self, data):
        reformattedData = np.array(data).reshape(1, 16, 16, 1)
        return self.vit.predict(reformattedData)