import sys
sys.path.append('../')

import torch
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

PARAMETER_TYPES = [WEIGHT, BIAS]
STATISTICS = [MEAN, VARIANCE]

class ParametersDistribution:
    def __init__(self, sharedWeightDim, headWeightDim, headCount):
        self.shared = {}
        self.hidden = {}
        self.headCount = headCount
        sharedBiasDim = tuple(sharedWeightDim[:-1])
        headBiasDim = tuple(headWeightDim[:-1])
        for parameterType in PARAMETER_TYPES:
            self.shared[parameterType] = {}
            self.hidden[parameterType] = {}
            for statistic in STATISTICS:
                sharedDim = sharedWeightDim if parameterType == WEIGHT else sharedBiasDim
                headDim = headWeightDim if parameterType == WEIGHT else headBiasDim
                self.shared[parameterType][statistic] = \
                    torch.rand(sharedDim).type(FloatTensor)
                self.hidden[parameterType][statistic] = {}
                for head in range(headCount):
                    self.hidden[parameterType][statistic][head] = \
                        torch.rand(headDim).type(FloatTensor)

    def getShared(self, parameterType, statistic):
        return self.shared[parameterType][statistic]

    def getHead(self, parameterType, statistic, head):
        return self.hidden[parameterType][statistic][head]

    def overwrite(self, q):
        for parameterType in PARAMETER_TYPES:
            for statistic in STATISTICS:
                self.shared[parameterType][statistic] = \
                    q.shared[parameterType][statistic]
                for head in range(self.headCount):
                    self.hidden[parameterType][statistic][head] = \
                        q.hidden[parameterType][statistic][head]
