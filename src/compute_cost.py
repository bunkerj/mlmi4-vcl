import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS
from monte_carlo import MonteCarlo
from KL import KL

def computeCost(model, inputs, labels, qPos, qPri, taskId):
    monteCarlo = MonteCarlo(model)
    kl = KL()
    cost = - (monteCarlo.logPred(inputs, labels, qPos, taskId, inputs.size()[0]) - torch.div(kl.computeKL(qPos, qPri, taskId), 10)) 

    return  cost
