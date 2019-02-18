import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def onehot(labels, num_classes):
    y_onehot = labels.numpy()
    y_onehot = (np.arange(num_classes) == y_onehot[:,None]).astype(np.float32)
    y_onehot = torch.from_numpy(y_onehot)#.to(device)
    return y_onehot

class MnistGen():
    def __init__(self):
        # Define number of tasks
        self.maxIter = 1
        self.curIter = 1
        self.X_train = torch.load('../data/MNIST_X_train.pt')
        self.Y_train = torch.load('../data/MNIST_Y_train.pt')
        self.X_test = torch.load('../data/MNIST_X_test.pt')
        self.Y_test = torch.load('../data/MNIST_Y_test.pt')

    def get_dims(self):
        return 784, 1

    def next_task(self):
        if self.curIter > self.maxIter:
            raise Exception('Task finished!')
        else:
            self.curIter += 1

            next_x_train = self.X_train
            next_y_train = onehot(self.Y_train, 10)

            next_x_test = self.X_test
            nex_y_test = onehot(self.Y_test, 10)

            return nex_x_train, next_y_train, next_x_test, next_y_test

class SplitMnistGen()
    def __init__(self):
        self.maxIter = 5
        self.curIter = 1
        self.X_train = torch.load('../data/MNIST_X_train.pt')
        self.Y_train = torch.load('../data/MNIST_Y_train.pt')
        self.X_test = torch.load('../data/MNIST_X_test.pt')
        self.Y_test = torch.load('../data/MNIST_Y_test.pt')

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]

    def change_ordering(self, set0, set1):
        self.sets_0 = set0
        self.sets_1 = set1

    def get_dims(self):
        return 784, 2

    def next_task(self):
        if self.curIter > self.maxIter:
            raise Exception('Task finished!')
        else:
            idx = (self.Y_train == 2)

            train_id_0 = X_train[self.Y_train == self.sets_0[self.curIter], :, :, :]
            train_id_1 = X_train[self.Y_train == self.sets_1[self.curIter], :, :, :]
            next_x_train = torch.cat([train_id_0, train_id_1], dim=0)

            next_y_train = torch.cat([torch.ones(train_id_0.shape[0], 1),torch.zeros(train_id_1.shape[0], 1)],dim=0)
            next_y_train = torch.cat([next_y_train, 1-next_y_train])

            test_id_0 = X_train[self.Y_test == self.sets_0[self.curIter], :, :, :]
            test_id_1 = X_train[self.Y_test == self.sets_1[self.curIter], :, :, :]
            next_x_test = torch.cat([test_id_0, test_id_1], dim=0)

            next_y_test = torch.cat([torch.ones(test_id_0.shape[0], 1),torch.zeros(test_id_1.shape[0], 1)],dim=0)
            next_y_test = torch.cat([next_y_test, 1-next_y_test])

            self.curIter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
