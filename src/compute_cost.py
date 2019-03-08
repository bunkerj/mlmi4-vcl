import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS
from monte_carlo import MonteCarlo
from KL import KL

def computeCost(inputs, labels, qPos, qPri, taskId, numSamples, alpha=1):
    inputSize = inputs.size()[0]
    monteCarlo = MonteCarlo(qPos, numSamples)
    kl = KL()
    mcTerm = monteCarlo.logPred(inputs, labels, taskId)
    klTerm = torch.div(kl.computeKL(qPos, qPri, taskId), inputSize)
    return -((2-alpha)*mcTerm - alpha*klTerm)
