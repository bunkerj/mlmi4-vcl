import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from test_nn import TestNN

# device configuration
device = torch.device('cuda')
# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
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

# Fully Connected neural network with one hidden layer
model = TestNN(input_size, hidden_size, num_classes).to(device)

# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Train the model

def _onehot(labels):
    y_onehot = labels.numpy()
    y_onehot = (np.arange(num_classes) == y_onehot[:,None]).astype(np.float32)
    y_onehot = torch.from_numpy(y_onehot).to(device)
    return y_onehot

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = _onehot(labels)

        # Forward pass
        output = model(images)
        loss = model.loss(output, labels)

        # backwards and optimize (to get loss value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0;
    total = 0;

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
