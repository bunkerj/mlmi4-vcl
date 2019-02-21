from parameters_distribution import ParametersDistribution
from data_gen import *
from coreset import *
from optimizer import minimizeLoss

#sharedDim = (3, 3, 3)
#headDim = (2, 3, 1)
#headCount = 3
#sharedDim, headDim, headCount,

# Specify the experiment setting
# 1. Data generator - MnistGen() / SplitMnistGen() / PermutedMnistGen()
#                   - NotMnistGen() / SplitNotMnistGen() / PermutedNotMnistGen()
dataGen = SplitMnistGen()
# 2. Coreset method and size - coreset_rand / coreset_k
coresetMethod = coreset_rand
coresetSize = 20

hidden_size,
no_epochs,
data_gen,
coreset_method,
coreset_size=0,
batch_size=None,
single_head=True

class VariationalTrainer:
    def __init__(self, noEpochs, batchSize, dataGen, coresetMethod, coresetSize):
        self.qPosterior = ParametersDistribution()

        self.dataGen = dataGen
        self.coresetMethod = coresetMethod
        self.coresetSize = coresetSize

        self.noEpochs = noEpochs
        self.batchSize = batchSize

    def train():

        # initialize current iteration
        self.dataGen.curIter = 0
        # initialize coreset
        x_coreset = torch.FloatTensor()
        y_coreset = torch.ByteTensor()
        # initialize x_testsets
        x_testsets = torch.FloatTensor()
        y_testsets = torch.ByteTensor()

        # obtain input and output dimension
        input_dim, output_dim = self.dataGen.get_dims()
        # obtain the number of tasks
        no_tasks = self.dataGen.maxIter

        for task_id in range(no_tasks):
            # train and test data for current task
            x_train, y_train, x_test, y_test = self.dataGen.next_task()
            # append test data to test sets
            x_testsets = torch.cat([x_testsets, x_test], dim=0)
            y_testsets = torch.cat([x_testsets, x_test], dim=0)

            # initialize the network with maximum likelihood weights
            if task_id == 0:
                model = model() #specify model
                model.train(x_train, y_train, no_epochs, batch_size)
                mf_weights = model.get_weights()
                mf_variances = None

            # create coreset
            if self.coresetSize > 0:
                x_coreset, y_coreset, x_train, y_train = self.coresetMethod(x_coreset, y_coreset, x_train, y_train, self.coresetSize)

            # train using mfvi nn
            pass #

            qPosterior.overwrite(maximizeVariationalLowerBound(qPosterior, next_y_train))

            testAccuracy(next_x_train, qPosterior, coreset)

    def self.getBatch(x_train, y_train):
        self.batchSize
        batches = []
        numberOfBatches = x_train.size() / self.batchSize
        if isinstance(numberOfBatches, int):
            errMessage = 'Batch size {} not consistent with dataset size {}' \
                .format(x_train.size(), self.batchSize)
            raise Exception(errMessage)
        for i in range(numberOfBatches):
            start = i*self.batchSize
            end = (i+1)*self.batchSize
            x_train_batch = x_train[start:end]
            y_train_batch = y_train[start:end]
            batches.append((x_train_batch, y_train_batch))
        return batches

    def maximizeVariationalLowerBound(x_train_batch, y_train_batch, qPrior, taskId):
        qPosterior = ParametersDistribution()
        parameters = qPosterior.getFlattenedParameters(taskId)
        optimizer = torch.optim.Adam(parameters, lr = 0.001)
        for x_train_batch, y_train_batch in self.getBatch(x_train, y_train):
            lossArgs = (x_train_batch, y_train_batch qPosterior, qPrior, taskId)
            minimizeLoss(1000, optimizer, getCost, lossArgs)
        return qPosterior
