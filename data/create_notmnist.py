################################################################################
# Load & Save NotMnist Dataset as Pickle Files #################################
################################################################################

import os
import urllib.request
from scipy.io import loadmat
import numpy as np
import torch

# create necessary directories if they do not exist
dirs = ['raw', 'processed', 'raw/not_mnist', 'processed/not_mnist']
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# if data does not exist, download from the web
if not os.path.exists('raw/not_mnist/notMNIST_small.mat'):
    print('Downloading NotMnist dataset...')
    url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat'
    urllib.request.urlretrieve(url, 'raw/not_mnist/notMNIST_small.mat')
    print('Download complete!')

# load the data (it is in dictionary type)
data = loadmat('raw/not_mnist/notMNIST_small.mat')

# extract images & labels
X = data['images']
Y = data['labels']

# transform numpy arrays into tensors
X = torch.from_numpy(X) / 255.0 #normalize
Y = torch.from_numpy(Y).type(torch.int64) #change dtype (necessary for one-hot-encoding)

# dimension of X: (28, 28, 18724)
# reshape into (28*28, 18724) & transpose -> (18724, 28*28)
X = torch.transpose(X.reshape(28*28, -1), 0, 1)

# for reproducibility
torch.manual_seed(0)

# in the data, images of same character are stacked together -> shuffle
# idx is the shuffled index & X.shape[0] is the number of images
idx = torch.randperm(X.shape[0])

# 90% of the data will be used as training data
n_train = int(X.shape[0]*0.9)

X_train = X[idx[:n_train],:]
Y_train = Y[idx[:n_train]]
X_test = X[idx[n_train:],:]
Y_test = Y[idx[n_train:]]

# save tensors as pickle file
torch.save(X_train, 'processed/not_mnist/NotMNIST_X_train.pt')
torch.save(Y_train, 'processed/not_mnist/NotMNIST_Y_train.pt')
torch.save(X_test, 'processed/not_mnist/NotMNIST_X_test.pt')
torch.save(Y_test, 'processed/not_mnist/NotMNIST_Y_test.pt')
