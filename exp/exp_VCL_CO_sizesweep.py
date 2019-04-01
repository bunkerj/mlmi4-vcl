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

directory = "../exp/final_experiments/VCL_CO_sizesweep"

dictUpdate = {
    'coresetMethod': coreset_rand,
    'coresetOnly': True,
    'numLayers' : (2,1),
    'numEpochs' : 120,
    'dataGen': SplitMnistGen()
}

for size in [40, 80, 160, 320, 640]:
    startTime = time()
    resultAverager = ResultAverager()
    dictUpdate['coresetSize'] = size
    dictParams = getAllExpParameters(dictUpdate)
    for iter in range(0,5):
        trainer = VariationalTrainer(dictParams)
        accuracy = trainer.train()
        resultAverager.add(accuracy)
    filename = str(size)
    writePerformanceRecordAccuracyAvg(directory, filename, resultAverager)
    print('Time for single iteration: {}'.format((time() - startTime) / 60))
