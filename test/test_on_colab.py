################################################################################
# Import Packages ##############################################################
################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
from scipy.io import loadmat
from scipy.stats import truncnorm

################################################################################
# Upload Files #################################################################
################################################################################

from google.colab import files
uploaded = files.upload()

################################################################################
# Constants ####################################################################
################################################################################

FloatTensor = (
    torch.cuda.FloatTensor
    if torch.cuda.is_available()
    else torch.FloatTensor)

Device = (
    torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu'))

MEAN = 'mean'
VARIANCE = 'variance'

WEIGHT = 'weight'
BIAS = 'bias'

INIT_VARIANCE = 0.05

################################################################################
# Variational Trainer ##########################################################
################################################################################

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

        self.coresetOnly = dictParams['coresetOnly']
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
        # initialize coresets & testsets
        x_coresets = {} ; y_coresets = {}
        x_testsets = {} ; y_testsets = {}
        for t in range(len(self.taskOrder)):
            taskId = self.taskOrder[t]
            headId = self.headOrder[t]
            # train and test data for current task
            self.dataGen.curIter = taskId
            x_train, y_train, x_testsets[taskId], y_testsets[taskId] = self.dataGen.next_task()
            # initialize the network with maximum likelihood weights
            if t == 0 and self.coresetOnly == False:
                self.modelInitialization(x_train, y_train, headId)
            # if coreset size is not zero and a new task is encountered, create coreset
            if self.coresetSize > 0 and taskId not in x_coresets.keys():
                x_coresets[taskId], y_coresets[taskId], x_train, y_train = self.coresetMethod(x_train, y_train, self.coresetSize)
            # update weights and bias for current task
            if self.coresetOnly == False:
                self.qPosterior.overwrite(self.maximizeVariationalLowerBound(self.qPosterior, x_train, y_train, headId))
            # get scores (this updates self.accuracy)
            self.getScores(x_coresets, y_coresets, x_testsets, y_testsets, t)
        return self.accuracy

    def modelInitialization(self, x_train, y_train, headId):
        model = VanillaNN(self.inputDim, self.hiddenSize, self.numSharedLayers+self.numHeadLayers, self.outputDim).to(Device)
        modelTrainer = NeuralTrainer(model)
        modelTrainer.train(x_train, y_train, None, self.numEpochs, self.batchSize, displayEpoch = 100)
        param_mean = model.getParameters()
        # use parameter mean to initialize the q prior
        self.qPosterior.setParameters(param_mean, headId)

    def getScores(self, x_coresets, y_coresets, x_testsets, y_testsets, t):
        for t_ in range(t+1):
            print("Getting scores... / Time: {}".format(t))
            taskId_ = self.taskOrder[t_]
            headId_ = self.headOrder[t_]
            if self.accuracy[taskId_][t] == 0:
                if self.numHeads == 1 and t_ == 0:
                    q_pred = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)
                    q_pred.overwrite(self.qPosterior)
                    if self.coresetSize > 0:
                        print("Updating q_pred with merged coreset...")
                        x_coreset, y_coreset = self.mergeCoresets(x_coresets, y_coresets)
                        q_pred.overwrite(self.maximizeVariationalLowerBound(q_pred, x_coreset, y_coreset, headId_))
                elif self.numHeads is not 1:
                    q_pred = ParametersDistribution(self.sharedWeightDim, self.headWeightDim, self.numHeads)
                    q_pred.overwrite(self.qPosterior)
                    if self.coresetSize > 0:
                        print("Updating q_pred with coreset... / Task ID: {}".format(taskId_))
                        q_pred.overwrite(self.maximizeVariationalLowerBound(q_pred, x_coresets[taskId_], y_coresets[taskId_], headId_))
                self.accuracy[taskId_][t] = self.testAccuracy(x_testsets[taskId_], y_testsets[taskId_], q_pred, headId_)
                print('Task ID: {} / Arrival Time: {} / Accuracy: {}'.format(taskId_, t, self.accuracy[taskId_][t]))

    def testAccuracy(self, x_test, y_test, q_pred, headId):
        acc = 0
        for x_test_batch, y_test_batch in self.getBatch(x_test, y_test):
            monteCarlo = MonteCarlo(q_pred, self.numSamples)
            y_pred_batch = monteCarlo.computeMonteCarlo(x_test_batch, headId)
            _, y_pred_batch = torch.max(y_pred_batch.data, 1)
            y_pred_batch = torch.eye(self.dataGen.get_dims()[1])[y_pred_batch].type(FloatTensor)
            acc += torch.sum(torch.mul(y_pred_batch, y_test_batch)).item()
        return acc / y_pred_batch.shape[0]

    def mergeCoresets(self, x_coresets, y_coresets):
        x_coresets_list = list(x_coresets.values())
        y_coresets_list = list(y_coresets.values())
        merged_x = torch.cat(x_coresets_list, dim=0)
        merged_y = torch.cat(y_coresets_list, dim=0)
        return merged_x, merged_y

    def getNumBatches(self, x_train):
        batch_size = x_train.shape[0] if self.batchSize is None else self.batchSize
        return int(x_train.shape[0] / batch_size)

    def getBatch(self, x_train, y_train):
        batches = []
        for i in range(self.getNumBatches(x_train)):
            if self.batchSize == None:
                batches.append((x_train, y_train))
            else:
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
            for iter, train_batch in enumerate(self.getBatch(x_train, y_train)):
                x_train_batch, y_train_batch = train_batch
                lossArgs = (x_train_batch, y_train_batch, newPosterior, oldPosterior, headId, self.numSamples)
                loss = minimizeLoss(1, optimizer, computeCost, lossArgs)
                if iter % 100 == 0:
                    print('Max Variational ELBO: epoch: [{}/{}] and batch: [{}/{}]'.format(epoch+1, self.numEpochs, iter+1, self.getNumBatches(x_train)))
            print('Loss at epoch: {}'.format(epoch+1))
        return newPosterior

################################################################################
# Necessary Functions ##########################################################
################################################################################

def computeCost(inputs, labels, qPos, qPri, taskId, numSamples):
    inputSize = inputs.size()[0]
    monteCarlo = MonteCarlo(qPos, numSamples)
    kl = KL()
    return  - (monteCarlo.logPred(inputs, labels, taskId) - torch.div(kl.computeKL(qPos, qPri, taskId), inputSize))

class KL:
    def _getKL(self, m, v, m0, v0, parId):
        constTerm = ( - 0.5 * m.size()[0] * m.size()[1] if parId == WEIGHT
                        else -0.5 * m.size()[0])
        logStdDiff = torch.sum(torch.log(v0) - torch.log(v))
        muDiffTerm = 0.5 * torch.sum((v + (m0 - m)**2) / v0)
        return constTerm + logStdDiff + muDiffTerm

    def computeKL(self, qPos, qPri, taskId):
        kl = 0
        for layerId, m in enumerate(qPos.getShared(WEIGHT,MEAN)):
            v = qPos.getShared(WEIGHT, VARIANCE)[layerId]
            m0 = qPri.getShared(WEIGHT, MEAN)[layerId]
            v0 = qPri.getShared(WEIGHT, VARIANCE)[layerId]
            kl += self._getKL(m, v, m0, v0, WEIGHT)

        for layerId, m in enumerate(qPos.getShared(BIAS,MEAN)):
            v = qPos.getShared(BIAS, VARIANCE)[layerId]
            m0 = qPri.getShared(BIAS, MEAN)[layerId]
            v0 = qPri.getShared(BIAS, VARIANCE)[layerId]
            kl += self._getKL(m, v, m0, v0, BIAS)

        for layerId, m in enumerate(qPos.getHead(WEIGHT, MEAN, taskId)):
            v = qPos.getHead(WEIGHT, VARIANCE, taskId)[layerId]
            m0 = qPri.getHead(WEIGHT, MEAN, taskId)[layerId]
            v0 = qPri.getHead(WEIGHT, VARIANCE, taskId)[layerId]
            kl += self._getKL(m, v, m0, v0, WEIGHT)

        for layerId, m in enumerate(qPos.getHead(BIAS, MEAN, taskId)):
            v = qPos.getHead(BIAS, VARIANCE, taskId)[layerId]
            m0 = qPri.getHead(BIAS, MEAN, taskId)[layerId]
            v0 = qPri.getHead(BIAS, VARIANCE, taskId)[layerId]
            kl += self._getKL(m, v, m0, v0, BIAS)

        return kl

class MonteCarlo:

    def __init__(self, qPos, numSamples):
        self.numSamples = numSamples
        self.qPos = qPos

    def _computeParameters(self, m, v, eps):
        return torch.add(torch.mul(eps, v), m)

    def _getParameterDims(self, PARAMETER):
        return PARAMETER.size()

    def _getSampledParametersDims(self, PARAMETER):
        return ((self.numSamples, self._getParameterDims(PARAMETER[1])[0], self._getParameterDims(PARAMETER[1])[1])\
                    if PARAMETER[0] == 'weight' \
                        else (self.numSamples, self._getParameterDims(PARAMETER[1])[0]))

    def _createParameterSample(self, funcGetSpecificParameters, PARAMETER, layerId, taskId = None):
        m = funcGetSpecificParameters(PARAMETER, MEAN, taskId)[layerId]
        v = funcGetSpecificParameters(PARAMETER, VARIANCE, taskId)[layerId]
        eps = torch.randn(self._getSampledParametersDims((PARAMETER, m))).type(FloatTensor)
        return self._computeParameters(m, v, eps)

    def _forwardPass(self, inputs, weights, biases):
        act = inputs
        numLayers = len(weights)
        for i in range(numLayers):
            pred = torch.add(torch.matmul(act, weights[i]), biases[i])
            act = F.relu(pred)
        return pred

    def _loss(self, output, labels):
        loss = torch.sum(labels * F.log_softmax(output, -1), -1)
        return loss.mean()

    def computeMonteCarlo(self, inputs, taskId):

        weightSample, baisesSample = [], []
        for layerId, mW in enumerate(self.qPos.getShared(WEIGHT, MEAN)):
            weightSample.append(self._createParameterSample(self.qPos.getShared, WEIGHT, layerId))
            baisesSample.append(self._createParameterSample(self.qPos.getShared, BIAS, layerId))

        for layerId, mW in enumerate(self.qPos.getHead(WEIGHT, MEAN, taskId)):
            weightSample.append(self._createParameterSample(self.qPos.getHead, WEIGHT, layerId, taskId))
            baisesSample.append(self._createParameterSample(self.qPos.getHead, BIAS, layerId, taskId))
        return torch.sum(self._forwardPass(inputs, weightSample, baisesSample), dim = 0)/self.numSamples

    def logPred(self, inputs, labels, taskId):
        pred = self.computeMonteCarlo(inputs, taskId)
        logLik = self._loss(pred, labels)
        return logLik

class NeuralTrainer():
    def __init__(self, neuralNetwork):
        # Create random Tensors to hold input and output
        self.neuralNetwork = neuralNetwork

    def _assignOptimizer(self, learningRate = 0.001):
        self.train_step = torch.optim.Adam(self.neuralNetwork.parameters(), lr = learningRate)

    def train(self, xTrain, yTrain, taskId = 0, noEpochs=1000, batchSize=100, displayEpoch=5):
        N = xTrain.shape[0]
        if batchSize > N:
            batchSize = N
        # Training cycle
        costs = [];
        for epoch in range(noEpochs):
            permInds = list(range(xTrain.shape[0]))
            np.random.shuffle(permInds)
            curxTrain = xTrain[permInds]
            curyTrain = yTrain[permInds]

            avgCost = 0.
            totalBatch = int(np.ceil(N * 1.0 / batchSize))
            for i in range(totalBatch):
                startInd = i*batchSize
                endInd = np.min([(i+1)*batchSize, N])
                xBatch = curxTrain[startInd:endInd, :]
                yBatch = curyTrain[startInd:endInd, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                yPred = self.neuralNetwork(xBatch)
                loss = self.neuralNetwork.loss(yPred, yBatch)
                self._assignOptimizer()
                self.train_step.zero_grad()
                loss.backward()
                self.train_step.step()
                # Compute average loss
                avgCost += loss / totalBatch
                if (i+1) % displayEpoch == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, noEpochs, i+1, totalBatch, avgCost))
            costs.append(avgCost)
        print("Optimization Finished!")
        return costs

def minimizeLoss(maxIter, optimizer, lossFunc, lossFuncArgs):
    for i in range(maxIter):
        optimizer.zero_grad()
        loss = lossFunc(*lossFuncArgs)
        loss.backward(retain_graph = True)
        optimizer.step()
        if i % 100 == 0:
            print(loss)

PARAMETER_TYPES = [WEIGHT, BIAS]
STATISTICS = [MEAN, VARIANCE]

class ParametersDistribution:
    def __init__(self, sharedWeightDim, headWeightDim, headCount):
        """ dimension: (# layers, input size, output size) """
        self.shared = {}
        self.heads = {}
        self.headCount = headCount
        self.sharedLayerCount = sharedWeightDim[0]
        self.headLayerCount = headWeightDim[0]
        for parameterType in PARAMETER_TYPES:
            self.shared[parameterType] = {}
            self.heads[parameterType] = {}
            for statistic in STATISTICS:
                sharedSplitDim, headSplitDim = \
                    self.getSplitDimensions(sharedWeightDim, headWeightDim, parameterType)
                self.shared[parameterType][statistic] = sharedSplitDim
                self.heads[parameterType][statistic] = {}
                for head in range(headCount):
                    self.heads[parameterType][statistic][head] = headSplitDim

    def createTensor(self, dimension):
        return autograd \
            .Variable(torch.rand(dimension) \
            .type(FloatTensor), requires_grad=True)

    def splitByLayer(self, dimension):
        layerCount, inputSize, outputSize = dimension
        splittedLayers = []
        previousOutput = inputSize
        for layer in range(layerCount):
            splittedLayers.append(self.createTensor((previousOutput, outputSize)))
            previousOutput = outputSize
        return splittedLayers

    def getSplitDimensions(self, sharedWeightDim, headWeightDim, parameterType):
        if parameterType == WEIGHT:
            return self.splitByLayer(sharedWeightDim), \
                   self.splitByLayer(headWeightDim)
        else:
            return self.getSplitBiasDimensions(sharedWeightDim), \
                   self.getSplitBiasDimensions(headWeightDim)

    def getSplitBiasDimensions(self, dimension):
        layerCount, inputSize, outputSize = dimension
        splittedLayers = []
        for layer in range(layerCount):
            splittedLayers.append(self.createTensor((outputSize)))
        return splittedLayers

    def getShared(self, parameterType, statistic,  taskId = None): # Added taskId for convenience
        return self.shared[parameterType][statistic]

    def getHead(self, parameterType, statistic, head):
        return self.heads[parameterType][statistic][head]

    def getFlattenedParameters(self, head):
        return self.getShared(WEIGHT, MEAN) + \
        self.getShared(BIAS, MEAN) + \
        self.getHead(WEIGHT, MEAN, head) + \
        self.getHead(BIAS, MEAN, head)

    def getGenericListOfTensors(self, referenceList):
        tensorList = []
        for item in referenceList:
            variance = torch \
                .ones(item.size(), requires_grad=True) \
                .type(FloatTensor) * INIT_VARIANCE
            tensorList.append(variance)
        return tensorList

    def getInitialVariances(self, parameters):
        tensorList = self.getGenericListOfTensors(parameters)
        return (tensorList[:self.sharedLayerCount], \
                tensorList[self.sharedLayerCount:])

    def getInitialMeans(self, parameters):
        return (parameters[:self.sharedLayerCount], \
                parameters[self.sharedLayerCount:])

    def setParameters(self, parameters, taskId):
        weights, biases = parameters

        sharedWeightMeans, headWeightMeans = self.getInitialMeans(weights)
        self.shared[WEIGHT][MEAN] = sharedWeightMeans
        self.heads[WEIGHT][MEAN][taskId] = headWeightMeans

        sharedBiasMeans, headBiasMeans = self.getInitialMeans(biases)
        self.shared[BIAS][MEAN] = sharedBiasMeans
        self.heads[BIAS][MEAN][taskId] = headBiasMeans

        sharedWeightVariances, headWeightVariances = self.getInitialVariances(weights)
        self.shared[WEIGHT][VARIANCE] = sharedWeightVariances
        self.heads[WEIGHT][VARIANCE][taskId] = headWeightVariances

        sharedBiasVariances, headBiasVariances = self.getInitialVariances(biases)
        self.shared[BIAS][VARIANCE] = sharedBiasVariances
        self.heads[BIAS][VARIANCE][taskId] = headBiasVariances

    def purifyTensorList(self, tensorList):
        newTensorList = []
        for tensor in tensorList:
            tensor = tensor.detach()
            tensor.requires_grad = True
            newTensorList.append(tensor)
        return newTensorList

    def overwrite(self, q):
        for parameterType in PARAMETER_TYPES:
            for statistic in STATISTICS:
                self.shared[parameterType][statistic] = \
                    self.purifyTensorList(q.shared[parameterType][statistic])
                for head in range(self.headCount):
                    self.heads[parameterType][statistic][head] = \
                        self.purifyTensorList(q.heads[parameterType][statistic][head])

class VanillaNN(nn.Module):
    def __init__(self, netWorkInputSize, networkHiddenSize, numLayers, numClasses):
        super(VanillaNN, self).__init__()
        self.netWorkInputSize = netWorkInputSize
        self.networkHiddenSize = networkHiddenSize
        self.numLayers = numLayers
        self.numClasses = numClasses
        self.moduleList = nn.ModuleList(self._constructArchitecture())

    def _getLayerName(self, layerIndex):
        return 'fc{}'.format(layerIndex)

    def getLayerDimensions(self, layerIndex):
        inputSize = (
            self.netWorkInputSize
            if layerIndex == 0
            else self.networkHiddenSize)
        outputSize = (
            self.numClasses
            if layerIndex == (self.numLayers - 1)
            else self.networkHiddenSize)
        return outputSize, inputSize

    def _getLayerNames(self):
        return [self._getLayerName(layerIndex) for layerIndex in range(self.numLayers)]

    def _constructArchitecture(self):
        layerNames = self._getLayerNames()
        moduleList = []
        for layerIndex, layerName in enumerate(layerNames):
            outputSize, inputSize = self.getLayerDimensions(layerIndex)

            layer = nn.Linear(inputSize, outputSize)
            layer.weight = torch.nn.Parameter(self._getTruncatedNormal([outputSize, inputSize], 0.02))
            layer.bias = torch.nn.Parameter(self._getTruncatedNormal([outputSize], 0.02))
            setattr(VanillaNN, layerName, layer)
            moduleList.append(layer)
        return moduleList

    def forward(self, x):
        layerNames = self._getLayerNames()
        for layerIndex, layerName in enumerate(layerNames):
            if layerIndex != (self.numLayers - 1):
                layer = getattr(VanillaNN, layerName)
                x = layer(x)
                x = F.relu(x)
        lastLayerName = layerNames[-1]
        layer = getattr(VanillaNN, lastLayerName)
        x = layer(x)
        return x

    def _getTruncatedNormal(self, size, threshold=1):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        values = torch.from_numpy(values).type(FloatTensor)
        return values

    def loss(self, output, labels):
        loss = torch.sum(- labels * F.log_softmax(output, -1), -1)
        return loss.mean()

    def getParameters(self):
        return (
            [layer.weight.transpose(0, 1).detach() for layer in self.moduleList],
            [layer.bias.detach() for layer in self.moduleList])

    def noGrad(self, parameters):
        parameters.requires_grad = False
        return parameters

    def setParameters(self, parameters):
        weights, biases = parameters
        for index, layer in enumerate(self.moduleList):
            layer.weight = torch.nn.Parameter(weights[index].transpose(0, 1))
            layer.bias = torch.nn.Parameter(biases[index])

    def prediction(self, x_test):
        outputs = self.forward(x_test)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

################################################################################
# Coreset & Data Generator #####################################################
################################################################################

# Random Selection
def coreset_rand(x_train, y_train, coreset_size):
    # randomly permute the indices
    idx = torch.randperm(x_train.shape[0])

    x_coreset = x_train[idx[:coreset_size]]
    y_coreset = y_train[idx[:coreset_size]]

    # remaining indices form the training data
    x_train = x_train[idx[coreset_size:],:]
    y_train = y_train[idx[coreset_size:]]

    return x_coreset, y_coreset, x_train, y_train

# K-center Selection
def coreset_k(x_train, y_train, coreset_size):
    # distance tensor contains the distance to the furthest center, for each data point
    distance = torch.ones(x_train.shape[0]).type(FloatTensor)*float('inf')
    # randomly select the first center
    cur_idx = torch.randint(x_train.shape[0], (1,)).item()
    # list idx will contain the indices of the coreset

    idx = [cur_idx]
    for i in range(1, coreset_size):
        # subtract the image of current center from all other images
        # x_diff[s,:] is the difference between s'th image and c'th image (c = cur_idx)
        x_diff = x_train - x_train[cur_idx,:].expand(x_train.shape)
        # torch.norm(x_diff, dim=1) obtains the norm of each images
        # update the distance by selecting the shorter distance for each image
        distance = torch.min(distance, torch.norm(x_diff, dim=1))
        # the image with the highest distance is selected as the next center
        new_idx = torch.max(distance, 0)[1].item()
        idx.append(new_idx)
        # now, new_idx is the current center
        cur_idx = new_idx

    # at first timestep, x_coreset and y_coreset should be set to "torch.FloatTensor()"
    x_coreset = x_train[idx,:]
    y_coreset = y_train[idx]

    # idx_train: all the indices not in idx, used to update training data
    idx_train = [i for i in range(x_train.shape[0]) if i not in idx]

    # remaining indices form the training data
    x_train = x_train[idx_train,:]
    y_train = y_train[idx_train]

    return x_coreset, y_coreset, x_train, y_train

# Mnist Data Loader
class Mnist():
    def __init__(self):
        self.X_train = torch.load('MNIST_X_train.pt').type(FloatTensor)
        self.Y_train = torch.load('MNIST_Y_train.pt')
        self.X_test = torch.load('MNIST_X_test.pt').type(FloatTensor)
        self.Y_test = torch.load('MNIST_Y_test.pt')

# Mnist Generator (no split or permutation)
class MnistGen(Mnist):
    def __init__(self):
        super().__init__()
        self.maxIter = 1
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            next_x_train = self.X_train
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)
            next_x_test = self.X_test
            next_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Split Mnist Generator
class SplitMnistGen(Mnist):
    # use the original order unless specified
    def __init__(self, set0 = [0, 2, 4, 6, 8], set1 = [1, 3, 5, 7, 9]):
        super().__init__()
        self.maxIter = len(set0)
        self.curIter = 0
        self.set0 = set0
        self.set1 = set1

    def get_dims(self):
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            train_id_0 = self.X_train[self.Y_train == self.set0[self.curIter], :]
            train_id_1 = self.X_train[self.Y_train == self.set1[self.curIter], :]
            next_x_train = torch.cat([train_id_0, train_id_1], dim=0)

            next_y_train = torch.cat([torch.ones(train_id_0.shape[0], 1),torch.zeros(train_id_1.shape[0], 1)],dim=0)
            next_y_train = torch.cat([next_y_train, 1-next_y_train]).type(FloatTensor)

            test_id_0 = self.X_test[self.Y_test == self.set0[self.curIter], :]
            test_id_1 = self.X_test[self.Y_test == self.set1[self.curIter], :]
            next_x_test = torch.cat([test_id_0, test_id_1], dim=0)

            next_y_test = torch.cat([torch.ones(test_id_0.shape[0], 1),torch.zeros(test_id_1.shape[0], 1)],dim=0)
            next_y_test = torch.cat([next_y_test, 1-next_y_test]).type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Permuted Mnist Generator
class PermutedMnistGen(Mnist):
    def __init__(self, maxIter = 10):
        super().__init__()
        self.maxIter = maxIter
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            torch.manual_seed(self.curIter)
            idx = torch.randperm(self.X_train.shape[1])

            next_x_train = deepcopy(self.X_train)[:,idx]
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)

            next_x_test = deepcopy(self.X_test)[:,idx]
            next_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# NotMnist Data Loader
class NotMnist():
    def __init__(self):
        self.X_train = torch.load('NotMNIST_X_train.pt')
        self.Y_train = torch.load('NotMNIST_Y_train.pt')
        self.X_test = torch.load('NotMNIST_X_test.pt')
        self.Y_test = torch.load('NotMNIST_Y_test.pt')

# NotMnist Generator (no split or permutation)
class NotMnistGen(NotMnist):
    def __init__(self):
        super().__init__()
        self.maxIter = 1
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            next_x_train = self.X_train
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)
            next_x_test = self.X_test
            nex_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Split NotMnist Generator
class SplitNotMnistGen(NotMnist):
    # use the original order unless specified
    def __init__(self, set0 = ['A', 'B', 'C', 'D', 'E'], set1 = ['F', 'G', 'H', 'I', 'J']):
        super().__init__()
        self.maxIter = len(set0)
        self.curIter = 0
        self.sets_0 = list(map(lambda x: ord(x) - 65, set0))
        self.sets_1 = list(map(lambda x: ord(x) - 65, set1))

    def get_dims(self):
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            train_id_0 = self.X_train[self.Y_train == self.set0[self.curIter], :]
            train_id_1 = self.X_train[self.Y_train == self.set1[self.curIter], :]
            next_x_train = torch.cat([train_id_0, train_id_1], dim=0)

            next_y_train = torch.cat([torch.ones(train_id_0.shape[0], 1),torch.zeros(train_id_1.shape[0], 1)],dim=0)
            next_y_train = torch.cat([next_y_train, 1-next_y_train]).type(FloatTensor)

            test_id_0 = self.X_test[self.Y_test == self.set0[self.curIter], :]
            test_id_1 = self.X_test[self.Y_test == self.set1[self.curIter], :]
            next_x_test = torch.cat([test_id_0, test_id_1], dim=0)

            next_y_test = torch.cat([torch.ones(test_id_0.shape[0], 1),torch.zeros(test_id_1.shape[0], 1)],dim=0)
            next_y_test = torch.cat([next_y_test, 1-next_y_test]).type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Permuted NotMnist Generator
class PermutedNotMnistGen(NotMnist):
    def __init__(self, maxIter = 10):
        super().__init__()
        self.maxIter = maxIter
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            torch.manual_seed(self.curIter)
            idx = torch.randperm(self.X_train.shape[1])

            next_x_train = deepcopy(self.X_train)[:,idx]
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)

            next_x_test = deepcopy(self.X_test)[:,idx]
            next_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
