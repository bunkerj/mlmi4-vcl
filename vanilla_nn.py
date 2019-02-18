import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

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

    def _getLayerDimensions(self, layerIndex):
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
            outputSize, inputSize = self._getLayerDimensions(layerIndex)

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
        values = torch.from_numpy(values).type(torch.cuda.FloatTensor)
        return values

    def loss(self, output, labels):
        criterion = nn.MultiLabelSoftMarginLoss()
        return criterion(output, labels)

if __name__ == '__main__':
    net = VanillaNN(
        netWorkInputSize = 5,
        networkHiddenSize = 7,
        numLayers = 3,
        numClasses = 9)
    input = torch.randn(1, 5).type(torch.cuda.FloatTensor)
    out = net(input)
