import utils.PCAHelper as helper
import machineLearners.CNNHandler as cnnhandler

class CNNService:
    def __init__(self):
        self.CNNHandler = cnnhandler.CNNHandler()
        self.train("./data/PCA/USPS.mat")
    def train(self, path):
        data, ys = helper.load_data(path)
        self.CNNHandler.train(data, ys)
    def predict(self, data):
        return self.CNNHandler.predict(data)