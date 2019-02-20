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
        self.shared = {}
        self.hidden = {}
        self.headCount = headCount
        sharedBiasDim = (sharedWeightDim[0], sharedWeightDim[2])
        headBiasDim = (headWeightDim[0], headWeightDim[2])
        for parameterType in PARAMETER_TYPES:
            self.shared[parameterType] = {}
            self.hidden[parameterType] = {}
            for statistic in STATISTICS:
                sharedDim = sharedWeightDim if parameterType == WEIGHT else sharedBiasDim
                headDim = headWeightDim if parameterType == WEIGHT else headBiasDim
                self.shared[parameterType][statistic] = \
                    autograd.Variable(torch.rand(sharedDim).type(FloatTensor), requires_grad=True)
                self.hidden[parameterType][statistic] = {}
                for head in range(headCount):
                    self.hidden[parameterType][statistic][head] = \
                        autograd.Variable(torch.rand(headDim).type(FloatTensor), requires_grad=True)

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
