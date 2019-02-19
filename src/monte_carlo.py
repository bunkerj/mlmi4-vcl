import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import MEAN, VARIANCE, WEIGHT, BIAS


class MonteCarlo():

    def __init__(self, neuralNetwork):
        self.neuralNetwork = neuralNetwork

    def _computeParameters(m, v, eps):
        return torch.add(torch.multiply(eps, torch.exp(0.5*v)), m)

    def computeMonteCarlo(self, inputs, qPos, taskId, numSamples):
        act = inputs.unsqueeze(0).repeat(numSamples, 1, 1)
        for layerId, mW in enumerate(qPos.getShared(WEIGHT,MEAN)):
            vW = qPos.getShared(WEIGHT, VARIANCE)[layerId]
            mB = qPos.getShared(BIAS, MEAN)[layerId]
            vB = qPos.getShared(BIAS, VARIANCE)[layerId]

            epsW = torch.randn((numSamples, mW.size()[1], mW.size()[0]))
            epsB = torch.randn((numSamples, 1, mB.size()[0]))
            weights = _computeParameters(mW, vW, epsW)
            biases = _computeParameters(mB, vB, epsB)
            self.neuralNetwork.setParameters((weights, biases), layerId)
            act = self.neuralNetwork(act)

        for layerId, mW in enumerate(qPos.getHead(WEIGHT,MEAN)):
                vW = qPos.getHead(WEIGHT, VARIANCE)[layerId]
                mB = qPos.getHead(BIAS, MEAN)[layerId]
                vB = qPos.getHead(BIAS, VARIANCE)[layerId]

                epsW = torch.randn((numSamples, mW.size()[1], mW.size()[0]))
                epsB = torch.randn((numSamples, 1, mB.size()[0]))

                weights = _computeParameters(mW, vW, epsW)
                biases = _computeParameters(mB, vB, epsB)
                self.neuralNetwork.setParameters((weights, biases), layerId)
                act = self.neuralNetwork(act)
        return act  
