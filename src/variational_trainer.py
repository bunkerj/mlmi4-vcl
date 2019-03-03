from parameters_distribution import ParametersDistribution
from data_gen import *
from coreset import *
from optimizer import minimizeLoss
from copy import deepcopy
from vanilla_nn import VanillaNN
from neural_trainer import NeuralTrainer
from compute_cost import computeCost
from monte_carlo import MonteCarlo
from constants import Device
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

        # lastly, a dictionary to store accuracy
        self.accuracy = {}
        for taskId in self.taskOrder:
            self.accuracy[taskId] = [0]*len(self.taskOrder)

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
                model = VanillaNN(self.inputDim, self.hiddenSize, self.numSharedLayers+self.numHeadLayers, self.outputDim).to(Device)
                modelTrainer = NeuralTrainer(model)
                modelTrainer.train(x_train, y_train, None, self.numEpochs, self.batchSize, displayEpoch = 100)
                param_mean = model.getParameters()
                # use parameter mean to initialize the q prior
                self.qPosterior.setParameters(param_mean, headId)

            # if coreset size is not zero, create coreset
            if self.coresetSize > 0:
                # if a new task is encountered, create coreset
                if taskId not in x_coresets.keys():
                    x_coresets[taskId], y_coresets[taskId], x_train, y_train = self.coresetMethod(x_train, y_train, self.coresetSize)

            # update weights and bias for current task
            self.qPosterior.overwrite(self.maximizeVariationalLowerBound(self.qPosterior, x_train, y_train, headId))

            # qPred and inference
            for t_ in range(t+1):
                taskId_ = self.taskOrder[t_]
                headId_ = self.headOrder[t_]

                if self.accuracy[taskId_][t] == 0:
                    if self.numHeads == 1 and t_ == 0:
                        if len(x_coresets) > 0:
                            print("Coreset Time :-)")
                            q_pred = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)
                            q_pred.overwrite(self.qPosterior)
                            x_coreset, y_coreset = self.mergeCoresets(x_coresets, y_coresets)
                            q_pred.overwrite(self.maximizeVariationalLowerBound(q_pred, x_coreset, y_coreset, headId_))
                    elif self.numHeads is not 1:
                        if len(x_coresets) > 0:
                            q_pred = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)
                            q_pred.overwrite(self.qPosterior)
                            q_pred.overwrite(self.maximizeVariationalLowerBound(q_pred, x_coresets[taskId_], y_coresets[taskId_], headId_))

                    self.accuracy[taskId_][t] = self.testAccuracy(x_testsets[taskId_], y_testsets[taskId_], q_pred, headId_)
                    print('Accuracy of task: {} at time: {} is: {}'.format(taskId_, t, self.accuracy[taskId_][t].item()))

        return self.accuracy

    def testAccuracy(self, x_test, y_test, q_pred, headId):
        acc = 0
        print("Accuracy Time :-)")
        for x_test_batch, y_test_batch in self.getBatch(x_test, y_test):
            monteCarlo = MonteCarlo(q_pred, self.numSamples)
            y_pred_batch = monteCarlo.computeMonteCarlo(x_test_batch, headId)
            _, y_pred_batch = torch.max(y_pred_batch.data, 1)
            y_pred_batch = torch.eye(self.dataGen.get_dims()[1])[y_pred_batch].type(FloatTensor)
            acc += torch.sum(torch.mul(y_pred_batch, y_test_batch))

        return acc/y_pred_batch.shape[0]

    def mergeCoresets(self, x_coresets, y_coresets):
        x_coresets_list = list(x_coresets.values())
        y_coresets_list = list(y_coresets.values())
        merged_x = torch.cat(x_coresets_list, dim=0)
        merged_y = torch.cat(y_coresets_list, dim=0)
        return merged_x, merged_y

    def getBatch(self, x_train, y_train):
        self.batchSize
        batches = []
        numberOfBatches = x_train.shape[0] / self.batchSize
        if isinstance(numberOfBatches, int):
            errMessage = 'Batch size {} not consistent with dataset size {}' \
                .format(x_train.shape[0], self.batchSize)
            raise Exception(errMessage)
        for i in range(int(numberOfBatches)):
            start = i*self.batchSize
            end = (i+1)*self.batchSize
            x_train_batch = x_train[start:end]
            y_train_batch = y_train[start:end]
            batches.append((x_train_batch, y_train_batch))
        return batches

    def maximizeVariationalLowerBound(self, oldPosterior, x_train, y_train, headId):
        # create dummy new posterior
        newPosterior = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)
        newPosterior.overwrite(oldPosterior)

        parameters = newPosterior.getFlattenedParameters(headId)
        optimizer = torch.optim.Adam(parameters, lr = 0.001)
        for epoch in range(self.numEpochs):
            idx = torch.randperm(x_train.shape[0])
            x_train, y_train = x_train[idx], y_train[idx]
            for x_train_batch, y_train_batch in self.getBatch(x_train, y_train):
                lossArgs = (x_train_batch, y_train_batch, newPosterior, oldPosterior, headId, self.numSamples)
                minimizeLoss(1, optimizer, computeCost, lossArgs)
                break;
        return newPosterior
