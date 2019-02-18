import sys
sys.path.append('../')
sys.path.append('../src')

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from neural_trainer import NeuralTrainer
from vanilla_nn import VanillaNN
from constants import Device

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_layers = 3
batch_size = 100
learning_rate = 0.001

# Loading MNIST dataset
train_dataset = torchvision.datasets.MNIST(root = '../../data',
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)
test_dataset = torchvision.datasets.MNIST(root = '../../data',
                                            train = False,
                                            transform = transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = 600000,
                                            shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = 600000,
                                            shuffle = False)

def _onehot(labels):
    y_onehot = labels.numpy()
    y_onehot = (np.arange(num_classes) == y_onehot[:,None]).astype(np.float32)
    y_onehot = torch.from_numpy(y_onehot).to(Device)
    return y_onehot


for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(Device)
        y_onehot = _onehot(labels)
        vanillaNN = VanillaNN(input_size, hidden_size, num_layers, num_classes)
        neuralTrainer = NeuralTrainer(vanillaNN)
        neuralTrainer.train(images, y_onehot, no_epochs = 5, batch_size = 200, display_epoch = 20)
        parameters = vanillaNN.getParameters()
        vanillaNN.setParameters(parameters)

with torch.no_grad():
    correct = 0;
    total = 0;

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(Device)
        labels = labels.to(Device)
        outputs = vanillaNN(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
