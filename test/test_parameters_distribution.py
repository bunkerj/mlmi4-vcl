import sys
sys.path.append('../')
sys.path.append('../src')

import torch
import numpy as np
from vanilla_nn import VanillaNN
from parameters_distribution import ParametersDistribution
from constants import MEAN, VARIANCE, WEIGHT, BIAS

INITIAL_VAR = 1
LAYER_WEIGHTS = [7*torch.ones(2, 3), 5*torch.ones(2, 2), 3*torch.ones(1, 2)]
LAYER_BIASES = [7*torch.ones(2), 5*torch.ones(2), 3*torch.ones(1)]

def convertTensorToNumpyArray(tensor):
    return tensor.cpu().detach().numpy()

def areTensorsEqual(tensor1, tensor2):
    arr1 = convertTensorToNumpyArray(tensor1)
    arr2 = convertTensorToNumpyArray(tensor2)
    return np.array_equal(arr1, arr2)

# Test setParameters()

inputSize = 3
hiddenSize = 2
numClasses = 1
numLayers = 3

vanillaNN = VanillaNN(inputSize, hiddenSize, numLayers, numClasses)
vanillaNN.setParameters((LAYER_WEIGHTS, LAYER_BIASES))

sharedWeightDim = (numLayers-1, inputSize, hiddenSize)
headWeightDim = (1, hiddenSize, numClasses)
q = ParametersDistribution(sharedWeightDim, headWeightDim, 1)
q.setParameters(vanillaNN.getParameters(), 1)

# Test mean initialization

sharedLayerWeightMeans = q.getShared(WEIGHT, MEAN)
assert areTensorsEqual(sharedLayerWeightMeans[0], LAYER_WEIGHTS[0])
assert areTensorsEqual(sharedLayerWeightMeans[1], LAYER_WEIGHTS[1])

headLayerWeightMeans = q.getHead(WEIGHT, MEAN, 1)
assert areTensorsEqual(headLayerWeightMeans[0], LAYER_WEIGHTS[2])

# Test variance initialization

sharedLayerWeightVariances = q.getShared(WEIGHT, VARIANCE)
assert areTensorsEqual(sharedLayerWeightVariances[0], INITIAL_VAR * torch.ones(LAYER_WEIGHTS[0].size()))
assert areTensorsEqual(sharedLayerWeightVariances[1], INITIAL_VAR * torch.ones(LAYER_WEIGHTS[1].size()))

headLayerWeightVariances = q.getHead(WEIGHT, VARIANCE, 1)
assert areTensorsEqual(headLayerWeightVariances[0], INITIAL_VAR * torch.ones(LAYER_WEIGHTS[2].size()))

print('No errors! :)')

# Test overwrite()
