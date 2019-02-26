import sys
sys.path.append('../src/')

import torch
from optimizer import minimizeLoss

class TestNN(torch.nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.fc1 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def getParameters(self):
        return [self.fc1.weight.transpose(0, 1), self.fc1.bias]

    def setParameters(self, parameters):
        # Can anything be done here to stay connected to the graph?
        weight, bias = parameters
        self.fc1.weight = torch.nn.Parameter(weight.transpose(0, 1))
        self.fc1.bias = torch.nn.Parameter(bias)

def computeCost(parameters, input):
    testNN = TestNN()
    testNN.setParameters(parameters)
    cost = testNN(input) ** 2
    print(cost) # Cost stays the same :(
    return cost

input = torch.randn(1, 10)
weight = torch.ones((10, 1))
bias = torch.ones((1, 1))

parameters = (weight, bias)
lossArgs = (parameters, input)
optimizer = torch.optim.Adam(parameters, lr = 0.01)

minimizeLoss(10, optimizer, computeCost, lossArgs)
