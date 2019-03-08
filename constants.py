import torch

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

INIT_VARIANCE = -10
