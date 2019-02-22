import sys
sys.path.append('../src')

import torch
from vanilla_nn import VanillaNN
from parameters_distribution import ParametersDistribution
from constants import FloatTensor

HEAD_COUNT = 2
INPUT_SIZE = 3
HIDDEN_SIZE = 2
CLASS_COUNT = 1
LAYER_COUNT = 3
NUMBER_OF_SAMPLES = 2

sharedWeightDim = (LAYER_COUNT-1, INPUT_SIZE, HIDDEN_SIZE)
headWeightDim = (1, HIDDEN_SIZE, CLASS_COUNT)

vanillaNN = VanillaNN(INPUT_SIZE, HIDDEN_SIZE, LAYER_COUNT, CLASS_COUNT)
q = ParametersDistribution(sharedWeightDim, headWeightDim, HEAD_COUNT)

weightSamples = [
    torch.ones((INPUT_SIZE, HIDDEN_SIZE)).type(FloatTensor),
    torch.ones((HIDDEN_SIZE, HIDDEN_SIZE)).type(FloatTensor),
    torch.ones((HIDDEN_SIZE, CLASS_COUNT)).type(FloatTensor),
    ]

biasSamples = [
    torch.ones((HIDDEN_SIZE)).type(FloatTensor),
    torch.ones((HIDDEN_SIZE)).type(FloatTensor),
    torch.ones((CLASS_COUNT)).type(FloatTensor),
    ]

vanillaNN.setParameters((weightSamples, biasSamples))

input = torch.ones(5, INPUT_SIZE).type(FloatTensor)
y = vanillaNN(input)
print(y)
