import sys
sys.path.append('../src/')

import torch
from copy import deepcopy
from data_gen import *
from variational_trainer import VariationalTrainer

def printBatches(batches):
    batchLength = len(batches)
    print('---- Data ----')
    for i in range(batchLength):
        batch = batches[i]
        print(batch[0])
    print('---- Labels ----')
    for i in range(batchLength):
        batch = batches[i]
        print(batch[1])

dictParams = {
    'batchSize': None,
    'numEpochs' : 100,
    'alpha' : 1,
    'dataGen' : PermutedMnistGen(),
    'numTasks' : 10,
    'numHeads' : 1,
    'numLayers' : (1,1),
    'hiddenSize' : 100,
    'taskOrder' : [],
    'headOrder' : [],
    'coresetOnly': False,
    'coresetSize': 0,
    'coresetMethod': None,
}

x_train = torch.ones(10, 5)
y_train = torch.ones(10, 2)
for i in range(y_train.shape[0]):
    x_train[i,:] *= i
    y_train[i,:] *= i

# ----------------- batchSize: None ----------------- #

x_train1 = deepcopy(x_train)
y_train1 = deepcopy(y_train)
dictParams1 = deepcopy(dictParams)

dictParams1['batchSize'] = None

variationalTrainer1 = VariationalTrainer(dictParams1)
batches1 = variationalTrainer1.getBatch(x_train1, y_train1)

x_train_batch, y_train_batch = batches1[0]
assert len(batches1) == 1
assert torch.all(torch.eq(x_train1, x_train_batch))
assert torch.all(torch.eq(y_train1, y_train_batch))

# ----------------- batchSize: 3 ----------------- #

x_train2 = deepcopy(x_train)
y_train2 = deepcopy(y_train)
dictParams2 = deepcopy(dictParams)

dictParams2['batchSize'] = 3

variationalTrainer2 = VariationalTrainer(dictParams2)
batches2 = variationalTrainer2.getBatch(x_train2, y_train2)

assert len(batches2) == 4

assert batches2[0][0].shape[0] == 3
assert batches2[1][0].shape[0] == 3
assert batches2[2][0].shape[0] == 3
assert batches2[3][0].shape[0] == 1

for i in range(10):
    batchIndex = int(i / dictParams2['batchSize'])
    tensorRow = i % dictParams2['batchSize']
    assert batches2[batchIndex][0][tensorRow,0] == i

print('No errors :)')
