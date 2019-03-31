import sys
sys.path.append('../')
sys.path.append('../src/')

from time import time
from data_gen import *
from coreset import *
from utils import *
from constants import DEFAULT_PARAMETERS
from result_averager import ResultAverager
from variational_trainer import VariationalTrainer

directory = "../exp/final_experiments/VCL_coreset_random"

dictUpdate = {
    'coresetMethod': coreset_rand,
    'numLayers' : (2,1),
    'coresetSize': 40,
    'numEpochs' : 120
}

for idx, dataGen in enumerate([SplitMnistGen(), SplitNotMnistGen()]):
    startTime = time()
    resultAverager = ResultAverager()
    dictUpdate['dataGen'] = dataGen
    dictParams = getAllExpParameters(dictUpdate)
    for iter in range(0,5):
        trainer = VariationalTrainer(dictParams)
        accuracy = trainer.train()
        resultAverager.add(accuracy)
    filename = ('splitMNIST' if idx == 0 else 'SplitNotMnist')
    writePerformanceRecordAccuracyAvg(directory, filename, resultAverager)
    print('Time for single iteration: {}'.format((time() - startTime) / 60))
