import sys
sys.path.append('../')
sys.path.append('../src')

import torch
from vanilla_nn import VanillaNN
from optimizer import minimizeLoss
from parameters_distribution import ParametersDistribution

INPUT_SIZE = 3
HIDDEN_SIZE = 2
LAYER_COUNT = 3
CLASS_COUNT = 1
HEAD_COUNT = 2

def getMSE(tensor1, tensor2):
    return ((tensor1 - tensor2)**2).sum()

def computeCost(qPrior, qPosterior):
    p1 = qPosterior.getFlattenedParameters(1)
    p2 = qPrior.getFlattenedParameters(1)
    return -sum([getMSE(p1[i], p2[i]) for i in range(len(p1))])

def sumAllParameters(params):
    totalSum = 0
    for param in params:
        totalSum += param.sum()
    return totalSum

def printParameterInfo(qPrior, qPosterior, title):
    topBanner = '--------------- {} ---------------'.format(title)
    print('\n' + topBanner)
    print('Cost: {}'.format(computeCost(qPrior, qPosterior)))
    print('Sum for qPosterior: {}'.format(sumAllParameters(qPosterior.getFlattenedParameters(1))))
    print('Sum for qPrior: {}'.format(sumAllParameters(qPrior.getFlattenedParameters(1))))
    print('-'*len(topBanner) + '\n')

sharedWeightDim = (LAYER_COUNT-1, INPUT_SIZE, HIDDEN_SIZE)
headWeightDim = (1, HIDDEN_SIZE, CLASS_COUNT)

vanillaNN = VanillaNN(INPUT_SIZE, HIDDEN_SIZE, LAYER_COUNT, CLASS_COUNT)
qPrior = ParametersDistribution(sharedWeightDim, headWeightDim, HEAD_COUNT)
qPosterior = ParametersDistribution(sharedWeightDim, headWeightDim, HEAD_COUNT)
# qPosterior.overwrite(qPrior)

printParameterInfo(qPrior, qPosterior, 'Before')

parameters = qPosterior.getFlattenedParameters(1)
optimizer = torch.optim.Adam(parameters, lr = 0.01)

lossArgs = (qPosterior, qPrior)
minimizeLoss(1000, optimizer, computeCost, lossArgs)

printParameterInfo(qPrior, qPosterior, 'After')
