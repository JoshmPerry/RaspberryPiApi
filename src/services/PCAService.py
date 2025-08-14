import utils.PCAHelper as helper
import machineLearners.PCA as pca
import random

class PCAService:
    def __init__(self):
        self.pca = pca.PCA(60, center_data=False)
        self.train("./data/PCA/USPS.mat")
    def train(self, path):
        data, _ = helper.load_data(path)
        self.pca.train(data)
    def compress(self, data):
        return self.pca.test(data)
    def reconstruct(self, data):
        return self.pca.reconstruct(data)
    def compressThenReconstruct(self, data):
        #TODO: Remove
        data, _ = helper.load_data("./data/PCA/USPS.mat")
        return data[random.randint(0, len(data) - 1)]
        compressed = self.compress(data)
        return self.reconstruct(compressed)
