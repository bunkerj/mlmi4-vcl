import sys
import torch
sys.path.append('../')
sys.path.append('../src/')

FloatTensor = (
    torch.cuda.FloatTensor
    if torch.cuda.is_available()
    else torch.FloatTensor)

Device = (
    torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu'))

MEAN = 'mean'
VARIANCE = 'variance'

WEIGHT = 'weight'
BIAS = 'bias'

INIT_VARIANCE = -6

DEFAULT_PARAMETERS = {
'numEpochs' : 120,
'batchSize' : None,
'alpha' : 1,
'dataGen' : None,
'numTasks' : 5,
'numHeads' : 5,
'numLayers' : (2,1),
'hiddenSize' : 256,
'taskOrder' : [],
'headOrder' : [],
'coresetMethod': None,
'coresetOnly': False,
'coresetSize': 0
}
