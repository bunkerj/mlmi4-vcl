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
hiddenSize = 256
numLayers = 5
numClasses = 2
batchSize = 50
numSamples = 10

sharedDim = (2, inputSize, hiddenSize)
headDim = (3, hiddenSize, numClasses)
headCount = 5

fakeBatch = torch.randn(batchSize,inputSize).type(FloatTensor)
fakeTarget = torch.cat([torch.ones(25, 2), torch.zeros(25, 2)]).type(FloatTensor)

qPosterior = ParametersDistribution(sharedDim, headDim, headCount)
vanillaNN = VanillaNN(inputSize, hiddenSize, numLayers, numClasses)
monteCarlo = MonteCarlo(vanillaNN)
result = monteCarlo.computeMonteCarlo(fakeBatch, qPosterior, 1, numSamples)
print(result)
predictionProb = monteCarlo.logPred(fakeBatch, qPosterior, 1, numSamples, fakeTarget)
print(predictionProb)
