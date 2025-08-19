import utils.PCAHelper as dataReader
import utils.DataHelper as dataHelper
import machineLearners.cnn.CNNHandler as cnnhandler

class CNNService:
    def __init__(self):
        self.CNNHandler = cnnhandler.CNNHandler()
        self.train("./data/training/USPS.mat")
    def train(self, path):
        data, ys = dataReader.load_data(path)
        reformattedData = data.reshape(data.shape[0], 16, 16, 1)
        trainDataset, validationDataset = dataHelper.arrange_data(reformattedData, ys)
        self.CNNHandler.train(trainDataset, validationDataset)
    def predict(self, data):
        return self.CNNHandler.predict(data)