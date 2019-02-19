import sys
sys.path.append('../')
sys.path.append('../src/')

import torch.autograd as autograd

import torch
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS
from parameters_distribution import ParametersDistribution
from KL import KL

sharedDim = (3, 3, 3)
headDim = (2, 3, 3)
headCount = 3
qPrior = ParametersDistribution(sharedDim, headDim, headCount)
qPosterior = ParametersDistribution(sharedDim, headDim, headCount)

qPosterior.setSharedAutograd(WEIGHT, VARIANCE)
qPosterior.setSharedAutograd(WEIGHT, MEAN)
qPosterior.setSharedAutograd(BIAS, VARIANCE)
qPosterior.setSharedAutograd(BIAS, MEAN)

qPosterior.setHeadAutograd(WEIGHT, VARIANCE, 2)
qPosterior.setHeadAutograd(WEIGHT, MEAN, 2)
qPosterior.setHeadAutograd(BIAS, VARIANCE, 2)
qPosterior.setHeadAutograd(BIAS, MEAN, 2)

kl = KL()
print(kl.computeKL(2, qPrior, qPrior))
