import sys
sys.path.append('../')

import torch
import torch.autograd as autograd
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

PARAMETER_TYPES = [WEIGHT, BIAS]
STATISTICS = [MEAN, VARIANCE]

class ParametersDistribution:
    def __init__(self, sharedWeightDim, headWeightDim, headCount):
        """ dimension: (# layers, input size, output size) """
        self.shared = {}
        self.hidden = {}
        self.headCount = headCount
        for parameterType in PARAMETER_TYPES:
            self.shared[parameterType] = {}
            self.hidden[parameterType] = {}
            for statistic in STATISTICS:
                sharedSplitDim, headSplitDim = \
                    self.getSplitDimensions(sharedWeightDim, headWeightDim, parameterType)
                self.shared[parameterType][statistic] = sharedSplitDim
                self.hidden[parameterType][statistic] = {}
                for head in range(headCount):
                    self.hidden[parameterType][statistic][head] = headSplitDim

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

    def getShared(self, parameterType, statistic):
        return self.shared[parameterType][statistic]

    def getHead(self, parameterType, statistic, head):
        return self.hidden[parameterType][statistic][head]

    def getFlattenedParameters(self, head):
        splitLayers = [
            self.getShared(WEIGHT, MEAN),
            self.getShared(BIAS, MEAN),
            self.getHead(WEIGHT, MEAN, head),
            self.getHead(BIAS, MEAN, head)]
        unsplitLayers = []
        for splitLayer in splitLayers:
            unsplitLayers += splitLayer
        return unsplitLayers

    def overwrite(self, q):
        for parameterType in PARAMETER_TYPES:
            for statistic in STATISTICS:
                self.shared[parameterType][statistic] = \
                    q.shared[parameterType][statistic]
                for head in range(self.headCount):
                    self.hidden[parameterType][statistic][head] = \
                        q.hidden[parameterType][statistic][head]
