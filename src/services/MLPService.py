import utils.PCAHelper as dataReader
import utils.DataHelper as dataHelper
import machineLearners.pca.PCA as pca
import machineLearners.mlp.MLPHandler as mlphandler

class MLPService:
    def __init__(self):
        self.pca = pca.PCA(60, center_data=False)
        self.MLPHandler = mlphandler.MLPHandler()
        self.train("./data/training/USPS.mat")
    def train(self, path):
        Xs, Ys = dataReader.load_data(path)
        data, ys = dataHelper.shuffle(Xs, Ys)
        trainData, testData, trainLabels, testLabels = dataHelper.train_valid_split(data, ys, 0.8)
        self.pca.train(trainData)
        simplifiedTrainData = self.pca.test(trainData)
        simplifiedTestData = self.pca.test(testData)
        trainDataset = dataHelper.transform_to_dataset(simplifiedTrainData, trainLabels)
        validationDataset = dataHelper.transform_to_dataset(simplifiedTestData, testLabels)
        self.MLPHandler.train(trainDataset, validationDataset)
    def predict(self, data):
        simplifiedData = self.pca.test(data)
        return self.MLPHandler.predict(simplifiedData)