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
qPosterior1 = ParametersDistribution(sharedDim, headDim, headCount)
qPosterior2 = ParametersDistribution(sharedDim, headDim, headCount)

kl = KL()

parameters = qPosterior1.getFlattenedParameters(2)
optimizer = torch.optim.Adam(parameters, lr = 0.001)
lossArgs = (qPosterior1, qPrior, 2)
minimizeLoss(1000, optimizer, kl.computeKL, lossArgs)

print('\n--------- Change initialization ---------\n')

parameters = qPosterior2.getFlattenedParameters(2)
optimizer = torch.optim.Adam(parameters, lr = 0.001)
lossArgs = (qPosterior2, qPrior, 2)
minimizeLoss(1000, optimizer, kl.computeKL, lossArgs)
