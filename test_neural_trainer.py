import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from neural_trainer import NeuralTrainer
from test_nn import TestNN

# device configuration
device = torch.device('cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 1
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
                                            batch_size = batch_size,
                                            shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)


x_train, y_train = [], []
for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        print(images.size())
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        y_onehot = labels.numpy()
        y_onehot = (np.arange(num_classes) == y_onehot[:,None]).astype(np.float32)
        y_onehot = torch.from_numpy(y_onehot)
        print(y_onehot.shape)
        testNN = TestNN(input_size, hidden_size, num_classes)
        neuralTrainer = NeuralTrainer(testNN)
        neuralTrainer.train(images, y_onehot, batch_size = 25)
