import sys
sys.path.append('../src/')

import torch
from optimizer import minimizeLoss
from torch.autograd import Variable

def computeCost(parameters, input):
    weight, bias = parameters
    return torch.add(torch.matmul(input, weight), bias)

input = torch.randn(1, 10)
weight = Variable(torch.randn(10, 1), requires_grad = True)
bias = Variable(torch.randn(1, 1), requires_grad = True)

parameters = [weight, bias]
optimizer = torch.optim.Adam(parameters, lr = 1)
lossArgs = (parameters, input)

print(computeCost(parameters, input))
minimizeLoss(100, optimizer, computeCost, lossArgs)
print(computeCost(parameters, input))
