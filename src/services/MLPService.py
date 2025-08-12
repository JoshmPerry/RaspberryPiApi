import utils.PCAHelper as helper
import machineLearners.PCA as pca
import machineLearners.MLPHandler as mlphandler
import utils.ImageHelper as image_helper

class MLPService:
    def __init__(self):
        self.pca = pca.PCA(60, center_data=False)
        self.MLPHandler = mlphandler.MLPHandler()
        self.train("./data/PCA/USPS.mat")
    def train(self, path):
        data, ys = helper.load_data(path)
        #self.pca.train(data)
        #simplifiedData = self.pca.test(data)
        self.MLPHandler.train(data, ys)
    def predict(self, data):
        #simplifiedData = image_helper.preprocess_image(self.pca.test(data))
        return self.MLPHandler.predict(data)