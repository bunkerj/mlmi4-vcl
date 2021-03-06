import sys
sys.path.append('../')
sys.path.append('../src/')

import numpy as np
import torch
import torch.autograd as autograd
from KL import KL
from optimizer import minimizeLoss
from parameters_distribution import ParametersDistribution
from vanilla_nn import VanillaNN
from monte_carlo import MonteCarlo
from optimizer import minimizeLoss
from compute_cost import computeCost
from neural_trainer import NeuralTrainer
from constants import Device, FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS
from copy import deepcopy

import torchvision
import torchvision.transforms as transforms

inputSize = 784
hiddenSize = 256
numLayers = 3
numClasses = 10
numSamples = 100
batchSize = 100

sharedDim = (2, inputSize, hiddenSize)
headDim = (1, hiddenSize, numClasses)
headCount = 5

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
                                            batch_size = 100,
                                            shuffle = False)

def _onehot(labels):
    yOneHot = labels.numpy()
    yOneHot = (np.arange(numClasses) == yOneHot[:,None]).astype(np.float32)
    yOneHot = torch.from_numpy(yOneHot).to(Device)
    return yOneHot

def getBatch(x_train, y_train):
        batches = []
        numberOfBatches = x_train.size()[0] / batchSize
        if isinstance(numberOfBatches, int):
            errMessage = 'Batch size {} not consistent with dataset size {}' \
                .format(x_train.size(), batchSize)
            raise Exception(errMessage)
        for i in range(int(numberOfBatches)):
            start = i*batchSize
            end = (i+1)*batchSize
            x_train_batch = x_train[start:end]
            y_train_batch = y_train[start:end]
            batches.append((x_train_batch, y_train_batch))
        return batches

def maximizeVariationalLowerBound(x_train, y_train, qPrior, taskId):
        for x_train_batch, y_train_batch in getBatch(x_train, y_train):
            qPosterior = ParametersDistribution(sharedDim, headDim, headCount)
            qPosterior.overwrite(qPrior)
            parameters = qPosterior.getFlattenedParameters(taskId)
            optimizer = torch.optim.Adam(parameters, lr = 0.001)
            lossArgs = (x_train_batch, y_train_batch, qPosterior, qPrior, taskId, numSamples)
            minimizeLoss(1, optimizer, computeCost, lossArgs)
            qPrior.overwrite(qPosterior)
        return qPosterior

# Training the vanillaNN
for i, (images, labels) in enumerate(trainLoader):
    images = images.reshape(-1, 28*28).to(Device)
    yOnehot = _onehot(labels)
    vanillaNN = VanillaNN(inputSize, hiddenSize, numLayers, numClasses).to(Device)
    neuralTrainer = NeuralTrainer(vanillaNN)
    neuralTrainer.train(images, yOnehot, noEpochs = 5, batchSize = 200, displayEpoch = 20)

print("VanillaNN Trained!")

parameters = vanillaNN.getParameters()
qPrior = ParametersDistribution(sharedDim, headDim, headCount)
qPrior.setParameters(parameters, 1)
parameters = qPrior.getFlattenedParameters(1)

for i, (images, labels) in enumerate(trainLoader):
    images = images.reshape(-1, 28*28).to(Device)
    yOnehot = _onehot(labels)
    qPosterior = maximizeVariationalLowerBound(images, yOnehot, qPrior, taskId = 1)

print("Prediction Time :-) ")

with torch.no_grad():
    correct = 0;
    total = 0;

    for images, labels in testLoader:
        images = images.reshape(-1, 28*28).to(Device)
        labels = labels.to(Device)
        monteCarlo = MonteCarlo(qPrior, numSamples)
        predicted = monteCarlo.computeMonteCarlo(images, 1)
        _, predicted = torch.max(predicted.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
