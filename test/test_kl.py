import sys
sys.path.append('../')
sys.path.append('../src/')

import torch
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS
from parameters_distribution import ParametersDistribution
from KL import KL

sharedDim = (3, 3, 3)
headDim = (2, 3, 3)
headCount = 3
qPrior = ParametersDistribution(sharedDim, headDim, headCount)
qPosterior = ParametersDistribution(sharedDim, headDim, headCount)

kl = KL()
# qPosterior.overwrite(qPrior)
print(kl.computeKL(2, qPrior, qPrior))
