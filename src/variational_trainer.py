from parameters_distribution import ParametersDistribution
from data_gen import *
from coreset import *
from optimizer import minimizeLoss
from copy import deepcopy

# experiment setup
dictParams = {
'numEpochs':100,
'batchSize':10,
'dataGen':SplitMnistGen(),
'numTasks':5,
'numHeads':5,
'coresetMethod':coreset_rand,
'coresetSize':100,
'numLayers':(4,2),
'hiddenSize':100,
'taskOrder':[],
'headOrder':[],
}

# run experiment
VariationalTrainer = VariationalTrainer(dictParams)
acc = VariationalTrainer.train()
print(acc)

# variational trainer
class VariationalTrainer:
    def __init__(self, dictParams):
        """Parameters should be given as a dictionary,
        'numEpochs': number of epochs,
        'batchSize': batch size,
        'dataGen': data generator, options include MnistGen(), SplitMnistGen(),
                    PermutedMnistGen(), NotMnistGen(), SplitNotMnistGen(),
                    PermutedNotMnistGen()
        'numTasks': number of unique tasks
        'numHeads': number of network heads
        'coresetMethod': coreset heuristic, options include coreset_rand, coreset_k
        'coresetSize': coreset size
        'numLayers': number of shared and head layers, given as tuple
        'taskOrder' and 'headOrder': if specified, the 0th task in taskOrder will be
                                     fed to 0th head in headOrder, ...
        """

        self.numEpochs = dictParams['numEpochs']
        self.batchSize = dictParams['batchSize']

        self.dataGen = dictParams['dataGen']
        self.numTasks = dictParams['numTasks']
        self.numHeads = dictParams['numHeads']

        self.coresetSize = dictParams['coresetSize']
        if self.coresetSize > 0: self.coresetMethod = dictParams['coresetMethod']

        self.numSharedLayers, self.numHeadLayers = dictParams['numLayers']
        self.hiddenSize = dictParams['hiddenSize']
        self.inputDim, self.outputDim = self.dataGen.get_dims()
        sharedWeightDim = (self.numSharedLayers, self.inputDim, self.hiddenSize)
        headWeightDim = (self.numHeadLayers, self.hiddenSize, self.outputDim)
        self.qPosterior = ParametersDistribution(sharedWeightDim, headWeightDim, self.numHeads)

        # if number of tasks is same as number of heads,
        # 0th task -> 0th head, 1st task -> 1st head, ...
        if self.numTasks == self.numHeads:
            self.headOrder = [i for i in range(self.numHeads)]
            self.taskOrder = [i for i in range(self.numTasks)]
        # if the network is single head,
        # 0th task -> 0th head, 1st task -> 0th head, ...
        elif: self.numHeads == 1:
            self.headOrder = [0] * self.numTasks
            self.taskOrder = [i for i in range(self.numTasks)]
        # otherwise, use the given orders
        else:
            self.headOrder = dictParams['headOrder']
            self.taskOrder = dictParams['taskOrder']

        # lastly, a list to store
        self.accuracy = torch.zeros((self.numTasks,self.taskOrder))

    def train():
        # initialize coreset
        x_coresets = []
        y_coresets = []
        # initialize x_testsets
        x_testsets = []
        y_testsets = []

        for t in range(len(self.taskOrder)):
            # train and test data for current task
            self.dataGen.curIter = self.taskOrder[t]
            x_train, y_train, x_test, y_test = self.dataGen.next_task()
            # append test data to test sets
            x_testsets.append(x_test)
            y_testsets.append(y_test)

            # initialize the network with maximum likelihood weights
            if t == 0:
                model = VanillaNN(self.inputDim, self.hiddenSize, self.numSharedLayers+self.numHeadLayers, self.outputDim)
                modelTrainer = NeuralTrainer(model)
                modelTrainer.train(x_train, y_train, self.numEpochs, self.batchSize, displayEpoch = 20)
                param_mean = model.getParameters()
                # use parameter mean to initialize the q posterior
                self.qPosterior.setParameters(param_mean)

            # create coreset
            if self.coresetSize > 0:
                x_coresets, y_coresets, x_train, y_train = self.coresetMethod(x_coresets, y_coresets, x_train, y_train, self.coresetSize)

            # update weights and bias for current task
            self.qPosterior.overwrite(maximizeVariationalLowerBound(qPosterior, x_train, y_train))

            # qPred and inference
            for t_ in range(t):
                qPred = deepcopy(self.qPosterior)
                if self.numHeads == 1:
                    if len(x_coresets) > 0:
                        x_coreset, y_coreset = merge_coresets(x_coresets, y_coresets)
                        qPred.overwrite(maximizeVariationalLowerBound(qPred, x_coreset, y_coreset))
                else:
                    if len(x_coresets) > 0:
                        qPred.overwrite(maximizeVariationalLowerBound(qPred, x_coresets[t_], y_coresets[t_]))

                pass # make prediction for x_testsets[t_] and y_testsets[t_] and return accuracy

                self.accuracy(self.taskOrder[t_],t_) = acc

    def merge_coresets(x_coresets, y_coresets):
        merged_x, merged_y = x_coresets[0], y_coresets[0]
        for i in range(1, len(x_coresets)):
            merged_x = torch.cat([merged_x, x_coresets[i]], dim = 0)
            merged_y = torch.cat([merged_y, y_coresets[i]], dim = 0)
        return merged_x, merged_y

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
