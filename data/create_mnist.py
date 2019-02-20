################################################################################
# Load & Save MNIST Dataset as Pickle File #####################################
################################################################################

import torch
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(root = '../../data',
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)
test_dataset = torchvision.datasets.MNIST(root = '../../data',
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
    # change dtype of Y_test into integer (necessary for one-hot-encoding)
    Y_train = Y_train.type(torch.int64)
    # save X_train and Y_train as pickle file
    torch.save(X_train, 'MNIST_X_train.pt')
    torch.save(Y_train, 'MNIST_Y_train.pt')

for i, (X_test, Y_test) in enumerate(test_loader):
    # reshape X_test and Y_test as pickle file
    X_test = X_test.reshape(-1,28*28)
    # change dtype of Y_test into integer (necessary for one-hot-encoding)
    Y_test = Y_test.type(torch.int64)
    # save X_test and Y_test as pickle file
    torch.save(X_test, 'MNIST_X_test.pt')
    torch.save(Y_test, 'MNIST_Y_test.pt')
