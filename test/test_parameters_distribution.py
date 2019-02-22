import sys
sys.path.append('../')
sys.path.append('../src')

import torch
from util import *
from vanilla_nn import VanillaNN
from parameters_distribution import ParametersDistribution
from constants import MEAN, VARIANCE, WEIGHT, BIAS

INITIAL_VAR = 1
LAYER_WEIGHTS = [7*torch.ones(2, 3), 5*torch.ones(2, 2), 3*torch.ones(1, 2)]
LAYER_BIASES = [7*torch.ones(2), 5*torch.ones(2), 3*torch.ones(1)]

HEAD_COUNT = 2
INPUT_SIZE = 3
HIDDEN_SIZE = 2
CLASS_COUNT = 1
LAYER_COUNT = 3

sharedWeightDim = (LAYER_COUNT-1, INPUT_SIZE, HIDDEN_SIZE)
headWeightDim = (1, HIDDEN_SIZE, CLASS_COUNT)

# Test setParameters()

vanillaNN = VanillaNN(INPUT_SIZE, HIDDEN_SIZE, LAYER_COUNT, CLASS_COUNT)
vanillaNN.setParameters((LAYER_WEIGHTS, LAYER_BIASES))

q1 = ParametersDistribution(sharedWeightDim, headWeightDim, HEAD_COUNT)
q1.setParameters(vanillaNN.getParameters(), 1)

# Test mean initialization

sharedLayerWeightMeans = q1.getShared(WEIGHT, MEAN)
assert areTensorsEqual(sharedLayerWeightMeans[0], LAYER_WEIGHTS[0])
assert areTensorsEqual(sharedLayerWeightMeans[1], LAYER_WEIGHTS[1])

headLayerWeightMeans = q1.getHead(WEIGHT, MEAN, 1)
assert areTensorsEqual(headLayerWeightMeans[0], LAYER_WEIGHTS[2])

# Test variance initialization

sharedLayerWeightVariances = q1.getShared(WEIGHT, VARIANCE)
assert areTensorsEqual(sharedLayerWeightVariances[0], INITIAL_VAR * torch.ones(LAYER_WEIGHTS[0].size()))
assert areTensorsEqual(sharedLayerWeightVariances[1], INITIAL_VAR * torch.ones(LAYER_WEIGHTS[1].size()))

headLayerWeightVariances = q1.getHead(WEIGHT, VARIANCE, 1)
assert areTensorsEqual(headLayerWeightVariances[0], INITIAL_VAR * torch.ones(LAYER_WEIGHTS[2].size()))

# Test overwrite()

q2 = ParametersDistribution(sharedWeightDim, headWeightDim, HEAD_COUNT)
parameters1 = q1.getFlattenedParameters(1)
parameters2 = q2.getFlattenedParameters(1)
assert not areListOfTensorsEqual(parameters1, parameters2)

q2.overwrite(q1)
parameters1 = q1.getFlattenedParameters(1)
parameters2 = q2.getFlattenedParameters(1)
assert areListOfTensorsEqual(parameters1, parameters2)

print('No errors! :)')
