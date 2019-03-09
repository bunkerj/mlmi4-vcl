################################################################################
# Load & Save NotMnist Dataset as Pickle File ##################################
################################################################################

from scipy.io import loadmat
import numpy as np
import torch

# the data is dictionary type
data = loadmat('notMNIST_small.mat')

# extract images & labels
X = data['images']
Y = data['labels']

# Transform numpy arrays into tensors
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

# X is reshaped from (28, 28, 18724) to (28*28, 18724)
X = torch.transpose(X.reshape(28*28, -1), 0, 1)
# Divide X with 255.0 to normalise
X = X/255.0

# for reproducibility
torch.manual_seed(0)

# in the data, images of same character are stacked together
# so we need to shuffle them
# idx is the shuffled index
idx = torch.randperm(X.shape[0]) # X.shape[0] is the number of images

# 90% of the data will be used as training data
n_train = int(X.shape[0]*0.9)

X_train = X[idx[:n_train],:]
Y_train = Y[idx[:n_train]]
X_test = X[idx[n_train:],:]
Y_test = Y[idx[n_train:]]

# save tensors as pickle file
torch.save(X_train, 'NotMNIST_X_train.pt')
torch.save(Y_train, 'NotMNIST_Y_train.pt')
torch.save(X_test, 'NotMNIST_X_test.pt')
torch.save(Y_test, 'NotMNIST_Y_test.pt')
