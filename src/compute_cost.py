import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS
from monte_carlo import MonteCarlo
from KL import KL

def computeCost(inputs, labels, qPos, qPri, taskId, numSamples):
    monteCarlo = MonteCarlo()
    kl = KL()
    cost = monteCarlo.logPred(inputs, qPos, taskId, numSamples, labels) - torch.div(kl.computeKL(taskId, qPos, qPri), numSamples)

    return  cost
s
