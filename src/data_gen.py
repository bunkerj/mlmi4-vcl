import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
from scipy.io import loadmat
from constants import FloatTensor

torchvision.datasets.FashionMNIST(root='../data/fashion_mnist', train=True, download=True, transform=None)
torchvision.datasets.FashionMNIST(root='../data/fashion_mnist', train=False, download=True, transform=None)

# Mnist Data Loader
class Mnist():
    def __init__(self):
        self.X_train = torch.load('../data/MNIST_X_train.pt')
        self.Y_train = torch.load('../data/MNIST_Y_train.pt')
        self.X_test = torch.load('../data/MNIST_X_test.pt')
        self.Y_test = torch.load('../data/MNIST_Y_test.pt')

# Mnist Generator (no split or permutation)
class MnistGen(Mnist):
    def __init__(self):
        super().__init__()
        self.maxIter = 1
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            next_x_train = self.X_train
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)
            next_x_test = self.X_test
            next_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Split Mnist Generator
class SplitMnistGen(Mnist):
    # use the original order unless specified
    def __init__(self, set0 = [0, 2, 4, 6, 8], set1 = [1, 3, 5, 7, 9]):
        super().__init__()
        self.maxIter = len(set0)
        self.curIter = 0
        self.set0 = set0
        self.set1 = set1

    def get_dims(self):
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            train_id_0 = self.X_train[self.Y_train == self.set0[self.curIter], :]
            train_id_1 = self.X_train[self.Y_train == self.set1[self.curIter], :]
            next_x_train = torch.cat([train_id_0, train_id_1], dim=0).type(FloatTensor)

            next_y_train = torch.cat([torch.ones(train_id_0.shape[0], 1),torch.zeros(train_id_1.shape[0], 1)],dim=0)
            next_y_train = torch.cat([next_y_train, 1-next_y_train], dim=1).type(FloatTensor)

            test_id_0 = self.X_test[self.Y_test == self.set0[self.curIter], :]
            test_id_1 = self.X_test[self.Y_test == self.set1[self.curIter], :]
            next_x_test = torch.cat([test_id_0, test_id_1], dim=0).type(FloatTensor)

            next_y_test = torch.cat([torch.ones(test_id_0.shape[0], 1),torch.zeros(test_id_1.shape[0], 1)],dim=0)
            next_y_test = torch.cat([next_y_test, 1-next_y_test], dim=1).type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Permuted Mnist Generator
class PermutedMnistGen(Mnist):
    def __init__(self, maxIter = 10):
        super().__init__()
        self.maxIter = maxIter
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            torch.manual_seed(self.curIter)
            idx = torch.randperm(self.X_train.shape[1])

            next_x_train = deepcopy(self.X_train)[:,idx].type(FloatTensor)
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)

            next_x_test = deepcopy(self.X_test)[:,idx].type(FloatTensor)
            next_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# NotMnist Data Loader
class NotMnist():
    def __init__(self):
        self.X_train = torch.load('../data/NotMNIST_X_train.pt')
        self.Y_train = torch.load('../data/NotMNIST_Y_train.pt')
        self.X_test = torch.load('../data/NotMNIST_X_test.pt')
        self.Y_test = torch.load('../data/NotMNIST_Y_test.pt')

# NotMnist Generator (no split or permutation)
class NotMnistGen(NotMnist):
    def __init__(self):
        super().__init__()
        self.maxIter = 1
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            next_x_train = self.X_train
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)
            next_x_test = self.X_test
            nex_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Split NotMnist Generator
class SplitNotMnistGen(NotMnist):
    # use the original order unless specified
    def __init__(self, set0 = ['A', 'B', 'C', 'D', 'E'], set1 = ['F', 'G', 'H', 'I', 'J']):
        super().__init__()
        self.maxIter = len(set0)
        self.curIter = 0
        self.set0 = list(map(lambda x: ord(x) - 65, set0))
        self.set1 = list(map(lambda x: ord(x) - 65, set1))

    def get_dims(self):
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            train_id_0 = self.X_train[self.Y_train == self.set0[self.curIter], :]
            train_id_1 = self.X_train[self.Y_train == self.set1[self.curIter], :]
            next_x_train = torch.cat([train_id_0, train_id_1], dim=0).type(FloatTensor)

            next_y_train = torch.cat([torch.ones(train_id_0.shape[0], 1),torch.zeros(train_id_1.shape[0], 1)],dim=0)
            next_y_train = torch.cat([next_y_train, 1-next_y_train], dim=1).type(FloatTensor)

            test_id_0 = self.X_test[self.Y_test == self.set0[self.curIter], :]
            test_id_1 = self.X_test[self.Y_test == self.set1[self.curIter], :]
            next_x_test = torch.cat([test_id_0, test_id_1], dim=0).type(FloatTensor)

            next_y_test = torch.cat([torch.ones(test_id_0.shape[0], 1),torch.zeros(test_id_1.shape[0], 1)],dim=0)
            next_y_test = torch.cat([next_y_test, 1-next_y_test], dim=1).type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# Permuted NotMnist Generator
class PermutedNotMnistGen(NotMnist):
    def __init__(self, maxIter = 10):
        super().__init__()
        self.maxIter = maxIter
        self.curIter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            torch.manual_seed(self.curIter)
            idx = torch.randperm(self.X_train.shape[1])

            next_x_train = deepcopy(self.X_train)[:,idx]
            next_y_train = torch.eye(10)[self.Y_train].type(FloatTensor)

            next_x_test = deepcopy(self.X_test)[:,idx]
            next_y_test = torch.eye(10)[self.Y_test].type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

# FashionMnist Data Loader
class FashionMnist():
    def __init__(self):
        x_train, y_train = torch.load('../data/fashion_mnist/processed/training.pt')
        x_test, y_test = torch.load('../data/fashion_mnist/processed/test.pt')
        length = x_train.shape[0]
        self.X_train = x_train.view(x_train.shape[0], -1)
        self.Y_train = y_train
        self.X_test = x_test.view(x_test.shape[0], -1)
        self.Y_test = y_test

        plt.imshow(x_train[0].numpy().reshape(28,28), cmap = 'gray')

# FashionMnist Generator
class FashionMnistGen(FashionMnist):
    # use the original order unless specified
    def __init__(self, set0 = [0, 2, 4, 6, 8], set1 = [1, 3, 5, 7, 9]):
        super().__init__()
        self.maxIter = len(set0)
        self.curIter = 0
        self.set0 = set0
        self.set1 = set1

    def get_dims(self):
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.curIter >= self.maxIter:
            raise Exception('Task finished!')
        else:
            train_id_0 = self.X_train[self.Y_train == self.set0[self.curIter], :]
            train_id_1 = self.X_train[self.Y_train == self.set1[self.curIter], :]
            next_x_train = torch.cat([train_id_0, train_id_1], dim=0).type(FloatTensor)

            next_y_train = torch.cat([torch.ones(train_id_0.shape[0], 1),torch.zeros(train_id_1.shape[0], 1)],dim=0)
            next_y_train = torch.cat([next_y_train, 1-next_y_train], dim=1).type(FloatTensor)

            test_id_0 = self.X_test[self.Y_test == self.set0[self.curIter], :]
            test_id_1 = self.X_test[self.Y_test == self.set1[self.curIter], :]
            next_x_test = torch.cat([test_id_0, test_id_1], dim=0).type(FloatTensor)

            next_y_test = torch.cat([torch.ones(test_id_0.shape[0], 1),torch.zeros(test_id_1.shape[0], 1)],dim=0)
            next_y_test = torch.cat([next_y_test, 1-next_y_test], dim=1).type(FloatTensor)

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
