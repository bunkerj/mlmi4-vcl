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
    cost = monteCarlo.logPred(inputs, labels, qPos, taskId, numSamples) - torch.div(kl.computeKL(qPos, qPri, taskId), numSamples)

    return  cost
