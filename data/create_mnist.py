################################################################################
# Load & Save MNIST Dataset as Pickle Files ####################################
################################################################################

import os
import torch
import torchvision
import torchvision.transforms as transforms

# create necessary directories if they do not exist
dirs = ['raw', 'processed', 'raw/mnist', 'processed/mnist']
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# load dataset (download if necessary)
train_dataset = torchvision.datasets.MNIST(root = 'raw/mnist',
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)
test_dataset = torchvision.datasets.MNIST(root = 'raw/mnist',
                                            train = False,
                                            transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = 60000,
                                            shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = 10000,
                                            shuffle = False)

for i, (X_train, Y_train) in enumerate(train_loader):
    # reshape X_train
    X_train = X_train.reshape(-1,28*28)
    # save X_train and Y_train as pickle file
    torch.save(X_train, 'processed/mnist/MNIST_X_train.pt')
    torch.save(Y_train, 'processed/mnist/MNIST_Y_train.pt')

for i, (X_test, Y_test) in enumerate(test_loader):
    # reshape X_test
    X_test = X_test.reshape(-1,28*28)
    # save X_test and Y_test as pickle file
    torch.save(X_test, 'processed/mnist/MNIST_X_test.pt')
    torch.save(Y_test, 'processed/mnist/MNIST_Y_test.pt')
