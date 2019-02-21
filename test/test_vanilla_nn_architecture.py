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
inputSize = 784
hiddenSize = 500
numClasses = 10
numLayers = 3
learningRate = 0.001

# Loading MNIST dataset
trainDataset = torchvision.datasets.MNIST(root = '../../data',
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)
testDataset = torchvision.datasets.MNIST(root = '../../data',
                                            train = False,
                                            transform = transforms.ToTensor())

# Data Loader
trainLoader = torch.utils.data.DataLoader(dataset = trainDataset,
                                            batch_size = 600000,
                                            shuffle = True)
testLoader = torch.utils.data.DataLoader(dataset = testDataset,
                                            batch_size = 600000,
                                            shuffle = False)

def _onehot(labels):
    yOneHot = labels.numpy()
    yOneHot = (np.arange(numClasses) == yOneHot[:,None]).astype(np.float32)
    yOneHot = torch.from_numpy(yOneHot).to(Device)
    return yOneHot

for i, (images, labels) in enumerate(trainLoader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(Device)
        yOneHot = _onehot(labels)
        vanillaNN = VanillaNN(inputSize, hiddenSize, numLayers, numClasses)
        neuralTrainer = NeuralTrainer(vanillaNN)
        neuralTrainer.train(images, yOneHot, noEpochs = 5, batchSize = 200, displayEpoch = 20)
        parameters = vanillaNN.getParameters()
        vanillaNN.setParameters(parameters)

with torch.no_grad():
    correct = 0;
    total = 0;

    for images, labels in testLoader:
        images = images.reshape(-1, 28*28).to(Device)
        labels = labels.to(Device)
        predicted = vanillaNN.prediction(images)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
