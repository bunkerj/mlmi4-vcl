import sys
sys.path.append('../')
sys.path.append('../src/')

import torch
import torch.autograd as autograd
from KL import KL
from optimizer import minimizeLoss
from parameters_distribution import ParametersDistribution
from vanilla_nn import VanillaNN
from monte_carlo import MonteCarlo
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS

import torchvision
import torchvision.transforms as transforms

inputSize = 784
hiddenSize = 256
numLayers = 5
numClasses = 10
numSamples = 10

sharedDim = (2, inputSize, hiddenSize)
headDim = (3, hiddenSize, numClasses)
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
                                            batch_size = 100,
                                            shuffle = True)
testLoader = torch.utils.data.DataLoader(dataset = testDataset,
                                            batch_size = 100,
                                            shuffle = False)

def _onehot(labels):
    yOneHot = labels.numpy()
    yOneHot = (np.arange(numClasses) == yOneHot[:,None]).astype(np.float32)
    yOneHot = torch.from_numpy(yOneHot).to(Device)
    return yOneHot

for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        y_onehot = _onehot(labels)
        qPosterior = ParametersDistribution(sharedDim, headDim, headCount)
        vanillaNN = VanillaNN(inputSize, hiddenSize, numLayers, numClasses)
        monteCarlo = MonteCarlo(vanillaNN)

        result = monteCarlo.computeMonteCarlo(images, qPosterior, 1, numSamples)
        print(result)
        predictionProb = monteCarlo.logPred(images, qPosterior, 1, numSamples, y_onehot)
        print(predictionProb)
        break
