import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(VanillaNN, self).__init__()
        self.constructArchitecture(inputSize, hiddenSize, numClasses)

    def constructArchitecture(self, inputSize, hiddenSize, numClasses):
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc1.weight = torch.nn.Parameter(torch.zeros(hiddenSize,inputSize))
        self.fc1.bias = torch.nn.Parameter(torch.ones(hiddenSize))

        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc2.weight = torch.nn.Parameter(torch.zeros(hiddenSize,hiddenSize))
        self.fc2.bias = torch.nn.Parameter(torch.ones(hiddenSize))

        self.fc3 = nn.Linear(hiddenSize, numClasses)
        self.fc3.weight = torch.nn.Parameter(torch.zeros(numClasses,hiddenSize))
        self.fc3.bias = torch.nn.Parameter(torch.ones(numClasses))

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

net = VanillaNN(5, 7, 9)
input = torch.randn(1, 1, 1, 5)
print(net(input))
