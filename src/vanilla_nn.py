import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from constants import FloatTensor
from copy import deepcopy

class VanillaNN(nn.Module):
    def __init__(self, netWorkInputSize, networkHiddenSize, numLayers, numClasses):
        super(VanillaNN, self).__init__()
        self.netWorkInputSize = netWorkInputSize
        self.networkHiddenSize = networkHiddenSize
        self.numLayers = numLayers
        self.numClasses = numClasses
        self.moduleList = nn.ModuleList(self._constructArchitecture())

    def _getLayerName(self, layerIndex):
        return 'fc{}'.format(layerIndex)

    def getLayerDimensions(self, layerIndex):
        inputSize = (
            self.netWorkInputSize
            if layerIndex == 0
            else self.networkHiddenSize)
        outputSize = (
            self.numClasses
            if layerIndex == (self.numLayers - 1)
            else self.networkHiddenSize)
        return outputSize, inputSize

    def _getLayerNames(self):
        return [self._getLayerName(layerIndex) for layerIndex in range(self.numLayers)]

    def _constructArchitecture(self):
        layerNames = self._getLayerNames()
        moduleList = []
        for layerIndex, layerName in enumerate(layerNames):
            outputSize, inputSize = self.getLayerDimensions(layerIndex)

            layer = nn.Linear(inputSize, outputSize)
            layer.weight = torch.nn.Parameter(self._getTruncatedNormal([outputSize, inputSize], 0.02))
            layer.bias = torch.nn.Parameter(self._getTruncatedNormal([outputSize], 0.02))
            setattr(VanillaNN, layerName, layer)
            moduleList.append(layer)
        return moduleList

    def forward(self, x):
        layerNames = self._getLayerNames()
        for layerIndex, layerName in enumerate(layerNames):
            if layerIndex != (self.numLayers - 1):
                layer = getattr(VanillaNN, layerName)
                x = layer(x)
                x = F.relu(x)
        lastLayerName = layerNames[-1]
        layer = getattr(VanillaNN, lastLayerName)
        x = layer(x)
        return x

    def _getTruncatedNormal(self, size, threshold=1):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        values = torch.from_numpy(values).type(FloatTensor)
        return values

    def loss(self, output, labels):
        loss = torch.sum(- labels * F.log_softmax(output, -1), -1)
        return loss.mean()

    def getParameters(self):
        return (
            [layer.weight.transpose(0, 1).detach() for layer in self.moduleList],
            [layer.bias.detach() for layer in self.moduleList])

    def printArchitectureDimensions(self):
        weightDims = [layer.weight.transpose(0, 1).detach().size() for layer in self.moduleList]
        biasDims = [layer.bias.detach().size() for layer in self.moduleList]
        print('------ Network Architecture ------')
        print('Weights: {}'.format(weightDims))
        print('Bias: {}'.format(biasDims))
        print('----------------------------------')

    def noGrad(self, parameters):
        parameters.requires_grad = False
        return parameters

    def setParameters(self, parameters):
        weights, biases = parameters
        for index, layer in enumerate(self.moduleList):
            layer.weight = torch.nn.Parameter(weights[index].transpose(0, 1))
            layer.bias = torch.nn.Parameter(biases[index])

    def prediction(self, x_test):
        outputs = self.forward(x_test)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

if __name__ == '__main__':
    net = VanillaNN(
        netWorkInputSize = 5,
        networkHiddenSize = 7,
        numLayers = 3,
        numClasses = 9)
    input = torch.randn(1, 5).type(FloatTensor)
    out = net(input)
