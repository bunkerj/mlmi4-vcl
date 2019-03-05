import sys
sys.path.append('../')

import torch
import torch.autograd as autograd
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS, INIT_VARIANCE

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
            tensor = tensor.detach().clone()
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
