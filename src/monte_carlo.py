import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import Device, FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

class MonteCarlo:

    def __init__(self, qPos, numSamples):
        self.numSamples = numSamples
        self.qPos = qPos

    def _computeParameters(self, m, v, eps):
        return torch.add(torch.mul(eps, v), m)

    def _getParameterDims(self, PARAMETER):
        return PARAMETER.size()

    def _getSampledParametersDims(self, PARAMETER):
        return ((self.numSamples, self._getParameterDims(PARAMETER[1])[0], self._getParameterDims(PARAMETER[1])[1]) \
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
            # print(i)
            pred = torch.add(torch.matmul(act, weights[i]), biases[i])
            # print(pred.size())
            act = F.relu(pred)
        return pred

    def _loss(self, output, labels):
        loss = torch.sum(- labels * F.log_softmax(output, -1), -1)
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
