import torch
import torch.nn as nn
import torchvision
from scipy.stats import truncnorm
import torchvision.transforms as transforms


class TestNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(TestNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        size_weight = self.fc1.weight.size()
        size_bias = self.fc1.bias.size()
        self.fc1.weight = torch.nn.Parameter(self._truncated_normal(size_weight,0.02))
        self.fc1.bias = torch.nn.Parameter(self._truncated_normal(size_bias,0.02))
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        size_weight = self.fc2.weight.size()
        size_bias = self.fc2.bias.size()
        self.fc2.weight = torch.nn.Parameter(self._truncated_normal(size_weight,0.02))
        self.fc2.bias = torch.nn.Parameter(self._truncated_normal(size_bias,0.02))

        self.relu = nn.ReLU();
        self.fc3 = nn.Linear(hidden_size, num_classes)
        size_weight = self.fc3.weight.size()
        size_bias = self.fc3.bias.size()
        self.fc3.weight = torch.nn.Parameter(self._truncated_normal(size_weight,0.02))
        self.fc3.bias = torch.nn.Parameter(self._truncated_normal(size_bias,0.02))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def _truncated_normal(self, size, threshold=1):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        values = torch.from_numpy(values).type(torch.cuda.FloatTensor)
        return values

    def loss(output, labels): 
        criterion = nn.CrossEntropyLoss()
        return criterion(output, labels)
