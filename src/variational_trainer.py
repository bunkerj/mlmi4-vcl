from parameters_distribution import ParametersDistribution
from data_gen import *
from coreset import *
from optimizer import minimizeLoss
from copy import deepcopy
from vanilla_nn import VanillaNN
from neural_trainer import NeuralTrainer
from compute_cost import computeCost
from monte_carlo import MonteCarlo
import torch

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
        self.numSamples = dictParams['numSamples']

        self.dataGen = dictParams['dataGen']
        self.numTasks = dictParams['numTasks']
        self.numHeads = dictParams['numHeads']

        self.coresetSize = dictParams['coresetSize']
        if self.coresetSize > 0: self.coresetMethod = dictParams['coresetMethod']

        self.numSharedLayers, self.numHeadLayers = dictParams['numLayers']
        self.hiddenSize = dictParams['hiddenSize']
        self.inputDim, self.outputDim = self.dataGen.get_dims()
        self.sharedWeightDim = (self.numSharedLayers, self.inputDim, self.hiddenSize)
        self.headWeightDim = (self.numHeadLayers, self.hiddenSize, self.outputDim)
        self.qPosterior = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)

        # if number of tasks is same as number of heads,
        # 0th task -> 0th head, 1st task -> 1st head, ...
        if self.numTasks == self.numHeads:
            self.headOrder = list(range(self.numHeads))
            self.taskOrder = list(range(self.numTasks))
        # if the network is single head,
        # 0th task -> 0th head, 1st task -> 0th head, ...
        elif self.numHeads == 1:
            self.headOrder = [0] * self.numTasks
            self.taskOrder = list(range(self.numTasks))
        # otherwise, use the given orders
        else:
            self.headOrder = dictParams['headOrder']
            self.taskOrder = dictParams['taskOrder']

        # lastly, a list to store
        self.accuracy = {}
        for taskId in taskOrder:
            self.accuracy[taskId] = [0]*len(self.taskOrder)
        #torch.zeros((self.numTasks,len(self.taskOrder)))

    def train(self):
        # initialize coreset
        x_coresets = {}
        y_coresets = {}
        # initialize x_testsets
        x_testsets = {}
        y_testsets = {}

        for t in range(len(self.taskOrder)):
            taskId = self.taskOrder[t]
            headId = self.headOrder[t]
            # train and test data for current task
            self.dataGen.curIter = taskId
            x_train, y_train, x_testsets[taskId], y_testsets[taskId] = self.dataGen.next_task()

            # initialize the network with maximum likelihood weights
            if t == 0:
                model = VanillaNN(self.inputDim, self.hiddenSize, self.numSharedLayers+self.numHeadLayers, self.outputDim)
                modelTrainer = NeuralTrainer(model)
                modelTrainer.train(x_train, y_train, self.numEpochs, self.batchSize, displayEpoch = 20)
                param_mean = model.getParameters()
                # use parameter mean to initialize the q prior
                self.qPosterior.setParameters(param_mean)

            # if coreset size is not zero, create coreset
            if self.coresetSize > 0:
                # if a new task is encountered, create coreset
                if taskId not in x_coresets.keys():
                    x_coresets[taskId], y_coresets[taskId], x_train, y_train = self.coresetMethod(x_train, y_train, self.coresetSize)

            # update weights and bias for current task
            self.qPosterior.overwrite(maximizeVariationalLowerBound(self.qPosterior, x_train, y_train, headId))

            # qPred and inference
            for t_ in range(t):
                taskId_ = self.taskOrder[t_]
                headId_ = self.headOrder[t_]

                if self.accuracy[taskId_,t] == 0:
                    # overwrite qPosterior on q_pred
                    #qPred = deepcopy(self.qPosterior)
                    q_pred = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)
                    q_pred.overwrite(self.qPosterior)

                    if self.numHeads == 1:
                        if len(x_coresets) > 0:
                            x_coreset, y_coreset = merge_coresets(x_coresets, y_coresets)
                            q_pred.overwrite(maximizeVariationalLowerBound(q_pred, x_coreset, y_coreset, headId_))
                    else:
                        if len(x_coresets) > 0:
                            q_pred.overwrite(maximizeVariationalLowerBound(q_pred, x_coresets[taskId_], y_coresets[taskId_], headId_))

                    parameters = q_pred.parameters()
                    model = VanillaNN(self.inputDim, self.hiddenSize, self.numSharedlayers+self.numHeadLayers, self.outputDim)
                    monteCarlo = MonteCarlo(model)
                    y_pred = monteCarlo.computeMonteCarlo(x_testsets[taskId_], q_pred, headId_, self.numSamples)
                    _, y_pred = torch.max(y_pred.data, 1)

                    acc = torch.sum(torch.mul(y_pred, y_testsets[taskId_])) / y_pred.shape[0]

                    self.accuracy[taskId_,t] = acc

    def merge_coresets(self, x_coresets, y_coresets):
        x_coresets_list = list(x_coresets.values())
        y_coresets_list = list(y_coresets.values())
        merged_x = torch.cat(x_coresets_list, dim=0)
        merged_y = torch.cat(y_coresets_list, dim=0)
        return merged_x, merged_y

    def getBatch(self, x_train, y_train):
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

    def maximizeVariationalLowerBound(self, oldPosterior, x_train, y_train, headId):
        # create dummy new posterior
        newPosterior = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)
        newPoterior.overwrite(oldPosterior)

        parameters = newPostrior.getFlattenedParameters(headId)
        optimizer = torch.optim.Adam(parameters, lr = 0.001)

        for x_train_batch, y_train_batch in getBatch(x_train, y_train):
            lossArgs = (x_train_batch, y_train_batch, newPosterior, oldPosterior, headId)
            minimizeLoss(1000, optimizer, computeCost, lossArgs)

        return newPosterior

# experiment setup
dictParams = {
'numEpochs':100,
'batchSize':10,
'numSamples':10,
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
trainer = VariationalTrainer(dictParams)
trainer.train()
print(trainer.accuracy)
