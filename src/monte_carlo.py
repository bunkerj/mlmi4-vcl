import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import Device, FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

class MonteCarlo:

    def __init__(self, neuralNetwork):
        self.neuralNetwork = neuralNetwork

    def _computeParameters(self, m, v, eps):
        return torch.add(torch.mul(eps, v), m)

    def computeMonteCarlo(self, inputs, qPos, taskId, numSamples):

        weightSample, baisesSample = [], []
        for layerId, mW in enumerate(qPos.getShared(WEIGHT,MEAN)):
            vW = qPos.getShared(WEIGHT, VARIANCE)[layerId]
            mB = qPos.getShared(BIAS, MEAN)[layerId]
            vB = qPos.getShared(BIAS, VARIANCE)[layerId]

            epsW = torch.randn((numSamples, mW.size()[0], mW.size()[1])).type(FloatTensor)
            epsB = torch.randn((numSamples, mB.size()[0])).type(FloatTensor)
            weightSample.append(self._computeParameters(mW, vW, epsW))
            baisesSample.append(self._computeParameters(mB, vB, epsB))

        for layerId, mW in enumerate(qPos.getHead(WEIGHT,MEAN,taskId)):
            vW = qPos.getHead(WEIGHT, VARIANCE, taskId)[layerId]
            mB = qPos.getHead(BIAS, MEAN, taskId)[layerId]
            vB = qPos.getHead(BIAS, VARIANCE, taskId)[layerId]

            epsW = torch.randn((numSamples, mW.size()[0], mW.size()[1])).type(FloatTensor)
            epsB = torch.randn((numSamples, mB.size()[0])).type(FloatTensor)
            weightSample.append(self._computeParameters(mW, vW, epsW))
            baisesSample.append(self._computeParameters(mB, vB, epsB))

        pred = torch.zeros((inputs.size()[0], mW.size()[1])).type(FloatTensor)
        for k in range(numSamples):
            weights = [weight[k, :, :] for weight in weightSample]
            biases = [bias[k, :] for bias in baisesSample]
            self.neuralNetwork.setParameters((weights, biases)).to(Device)
            pred += self.neuralNetwork(inputs).to(Device)

        return pred/numSamples

    def logPred(self, inputs, labels, qPos, taskId, numSamples):
        pred = self.computeMonteCarlo(inputs, qPos, taskId, numSamples)
        logLik = self.neuralNetwork.loss(pred, labels)
        return logLik
