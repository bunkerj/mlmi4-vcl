import sys
sys.path.append('../')

import torch
import torch.autograd as autograd
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

PARAMETER_TYPES = [WEIGHT, BIAS]
STATISTICS = [MEAN, VARIANCE]

class ParametersDistribution:
    def __init__(self, sharedWeightDim, headWeightDim, headCount):
        # sharedWeightDim = (# layers, input, output)
        # headWeightDim = (# layers, input, output)
        self.shared = {}
        self.hidden = {}
        self.headCount = headCount
        for parameterType in PARAMETER_TYPES:
            self.shared[parameterType] = {}
            self.hidden[parameterType] = {}
            for statistic in STATISTICS:
                sharedDim, headDim = self.getDimensions(sharedWeightDim, headWeightDim, parameterType)
                self.shared[parameterType][statistic] = self.createTensor(sharedDim)
                self.hidden[parameterType][statistic] = {}
                for head in range(headCount):
                    self.hidden[parameterType][statistic][head] = self.createTensor(headDim)

    def createTensor(self, dimension):
        return autograd \
            .Variable(torch.rand(dimension) \
            .type(FloatTensor), requires_grad=True)

    def getDimensions(self, sharedWeightDim, headWeightDim, parameterType):
        if parameterType == WEIGHT:
            return sharedWeightDim, \
                   headWeightDim
        else:
            return self.getBiasDimensions(sharedWeightDim), \
                   self.getBiasDimensions(headWeightDim)

    def getBiasDimensions(self, weightDim):
        return (weightDim[0], weightDim[2])

    def getShared(self, parameterType, statistic):
        return self.shared[parameterType][statistic]

    def getHead(self, parameterType, statistic, head):
        return self.hidden[parameterType][statistic][head]

    def getFlattenedParameters(self, head):
        return [
            self.getShared(WEIGHT, MEAN),
            self.getShared(BIAS, MEAN),
            self.getHead(WEIGHT, MEAN, head),
            self.getHead(BIAS, MEAN, head)]

    def overwrite(self, q):
        for parameterType in PARAMETER_TYPES:
            for statistic in STATISTICS:
                self.shared[parameterType][statistic] = \
                    q.shared[parameterType][statistic]
                for head in range(self.headCount):
                    self.hidden[parameterType][statistic][head] = \
                        q.hidden[parameterType][statistic][head]
