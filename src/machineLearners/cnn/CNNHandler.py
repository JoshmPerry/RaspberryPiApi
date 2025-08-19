import machineLearners.cnn.CNN as cnn
import numpy as np

SAVE_MODEL_PATH = './data/models/cnn_model.keras'
REBUILD_MODEL = False

class CNNHandler:
    def __init__(self):
        self.cnn = cnn.CNN()

    def train(self, trainDataset, validationDataset):
        if REBUILD_MODEL:
            self.learn(trainDataset, validationDataset)
            return
        try:
            self.cnn.load_model(SAVE_MODEL_PATH)
        except Exception as e:  # Use Exception to catch all errors
            print("Error loading model:", e)
            self.learn(trainDataset, validationDataset)
    
    def learn(self, trainDataset, validationDataset):
        self.cnn.train(trainDataset, validationDataset)
        self.cnn.save_model(SAVE_MODEL_PATH)
        print("Model trained and saved.")

    def predict(self, data):
       reformattedData = np.array(data).reshape(1, 16, 16, 1)
       return self.cnn.predict(reformattedData)