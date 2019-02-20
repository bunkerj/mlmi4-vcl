import sys
sys.path.append('../')
sys.path.append('../src/')

import torch
import torch.autograd as autograd
from KL import KL
from optimizer import minimizeLoss
from parameters_distribution import ParametersDistribution
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

sharedDim = (3, 3, 3)
headDim = (2, 3, 1)
headCount = 3
qPrior = ParametersDistribution(sharedDim, headDim, headCount)
qPosterior = ParametersDistribution(sharedDim, headDim, headCount)

kl = KL()
parameters = qPosterior.getFlattenedParameters(2)
optimizer = torch.optim.Adam(parameters, lr = 0.001)
lossArgs = (2, qPosterior, qPrior)
minimizeLoss(10000, kl.computeKL, optimizer, lossArgs)
