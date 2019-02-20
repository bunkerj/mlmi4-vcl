from parameters_distribution import ParametersDistribution
from data_get import *

sharedDim = (3, 3, 3)
headDim = (2, 3, 1)
headCount = 3

CORESET_SIZE = 20

class VariationalTrainer:
    def __init__(self, sharedDim, headDim, headCount):
        self.qPosterior = ParametersDistribution()
        self.coreset = Coreset()
        self.dataset = SplitMnistGen()
        self.xCoreset = torch.FloatTensor()
        self.yCoreset = torch.FloatTensor()

    def train():
        for i in range(self.dataset.maxIter):
            next_x_train, next_y_train, next_x_test, next_y_test = self.dataset.next_task()
            xCoreset, yCoreset, next_x_train, next_y_train = coreset_rand(self.xCoreset, self.yCoreset, next_x_train, next_y_train, CORESET_SIZE)

            qPosterior.overwrite(maximizeVariationalLowerBound(qPosterior, next_y_train))
            
            testAccuracy(next_x_train, qPosterior, coreset)
