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

directory = "../exp/final_experiments/adversarial_ordering"

dictUpdate = {
    'dataGen':SplitMnistGen(),
    'coresetMethod': coreset_rand,
    'numLayers' : (2,1),
    'coresetSize': 0,
    'numEpochs' : 120,
    'taskOrder': list(range(5))
}

for headOrder in getHeadOrderList():
    startTime = time()
    resultAverager = ResultAverager()
    dictUpdate['headOrder'] = headOrder
    dictParams = getAllExpParameters(dictUpdate)
    for iter in range(0,5):
        trainer = VariationalTrainer(dictParams)
        accuracy = trainer.train()
        resultAverager.add(accuracy)
    filename = getName(dictParams, 'headOrder')
    writePerformanceRecordAccuracyAvg(directory, filename, resultAverager)
    print('Time for single iteration: {}'.format((time() - startTime) / 60))
