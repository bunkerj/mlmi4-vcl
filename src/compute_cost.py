import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS
from monte_carlo import MonteCarlo
from KL import KL

def computeCost(inputs, labels, qPos, qPri, taskId, numSamples):
    inputSize = inputs.size()[0]
    monteCarlo = MonteCarlo(qPos, numSamples)
    kl = KL()
    return  - (monteCarlo.logPred(inputs, labels, taskId) - torch.div(kl.computeKL(qPos, qPri, taskId), inputSize))
