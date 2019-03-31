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

directory = "../exp/final_experiments/head_sharing"

dictUpdate = {
    'dataGen':SplitMnistGen(),
    'coresetMethod': coreset_rand,
    'numLayers' : (2,1),
    'coresetSize': 0,
    'numEpochs' : 120,
    'taskOrder': list(range(5))
}

headCounts = [1,2]
taskCount = 5

for headCount in headCounts:
    for headOrder in getHeadOrderList(headCount, taskCount):
        startTime = time()
        resultAverager = ResultAverager()
        dictUpdate['headOrder'] = headOrder
        dictParams = getAllExpParameters(dictUpdate)
        for iter in range(0,5):
            trainer = VariationalTrainer(dictParams)
            accuracy = trainer.train()
            resultAverager.add(accuracy)
        filename = '{}_headcount_{}'.format(getName(dictParams, 'headOrder'), headCount)
        writePerformanceRecordAccuracyAvg(directory, filename, resultAverager)
        print('Time for single iteration: {}'.format((time() - startTime) / 60))
