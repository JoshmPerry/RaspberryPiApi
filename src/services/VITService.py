import utils.PCAHelper as dataReader
import utils.DataHelper as dataHelper
import machineLearners.vit.VITHandler as vithandler

class VITService:
    def __init__(self):
        self.VITHandler = vithandler.VITHandler()
        self.train("./data/training/USPS.mat")
    def train(self, path):
        data, ys = dataReader.load_data(path)
        reformattedData = data.reshape(data.shape[0], 16, 16, 1)
        trainDataset, validationDataset = dataHelper.arrange_data(reformattedData, ys)
        self.VITHandler.train(trainDataset, validationDataset)
    def predict(self, data):
        return self.VITHandler.predict(data)