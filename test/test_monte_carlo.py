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
hiddenSize = 500
numLayers = 3
numClasses = 2
batchSize = 200
numSamples = 100

sharedDim = (2, inputSize, hiddenSize)
headDim = (1, hiddenSize, numClasses)
headCount = 5

fakeBatch = torch.randn(batchSize,inputSize).type(FloatTensor)
qPosterior = ParametersDistribution(sharedDim, headDim, headCount)

# Wm = qPosterior.getShared(WEIGHT, MEAN)[0]
# print("Wm: {}".format(Wm.size()))
# Wv = qPosterior.getShared(WEIGHT, VARIANCE)[0]
# print("Wv: {}".format(Wv.size()))
# Bm = qPosterior.getShared(BIAS, MEAN)[0]
# print("Bm: {}".format(Bm.size()))
# Bv = qPosterior.getShared(BIAS, VARIANCE)[0]
# print("Bv: {}".format(Bv.size()))
#
# Wm = qPosterior.getShared(WEIGHT, MEAN)[1]
# print("Wm: {}".format(Wm.size()))
# Wv = qPosterior.getShared(WEIGHT, VARIANCE)[1]
# print("Wv: {}".format(Wv.size()))
# Bm = qPosterior.getShared(BIAS, MEAN)[1]
# print("Bm: {}".format(Bm.size()))
# Bv = qPosterior.getShared(BIAS, VARIANCE)[1]
# print("Bv: {}".format(Bv.size()))
#
# Wm = qPosterior.getHead(WEIGHT, MEAN, 1)[0]
# print("Wm: {}".format(Wm.size()))
# Wv = qPosterior.getHead(WEIGHT, VARIANCE, 1)[0]
# print("Wv: {}".format(Wv.size()))
# Bm = qPosterior.getHead(BIAS, MEAN, 1)[0]
# print("Bm: {}".format(Bm.size()))
# Bv = qPosterior.getHead(BIAS, VARIANCE, 1)[0]
# print("Bv: {}".format(Bv.size()))


vanillaNN = VanillaNN(inputSize, hiddenSize, numLayers, numClasses)
monteCarlo = MonteCarlo(vanillaNN)
result = monteCarlo.computeMonteCarlo(fakeBatch, qPosterior, 1, numSamples)
print(result)
