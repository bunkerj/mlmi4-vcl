import sys
sys.path.append('../')
sys.path.append('../src/')

import torch
import torch.autograd as autograd
from KL import KL
from optimizer import minimizeLoss
from parameters_distribution import ParametersDistribution
from vanilla_nn import VanillaNN
from monte_carlo import MonteCarlo
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

inputSize = 784
hiddenSize = 10
numLayers = 3
numClasses = 10
batchSize = 200
numSamples = 10

sharedDim = (2, inputSize, hiddenSize)
headDim = (1, hiddenSize, numClasses)
headCount = 5

fakeBatch = torch.randn(batchSize,inputSize).type(FloatTensor)
qPosterior = ParametersDistribution(sharedDim, headDim, headCount)
vanillaNN = VanillaNN(inputSize, hiddenSize, numLayers, numClasses)
monteCarlo = MonteCarlo(vanillaNN)
result = monteCarlo.computeMonteCarlo(fakeBatch, qPosterior, 1, numSamples)
print(result)
