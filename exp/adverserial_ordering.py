import sys
sys.path.append('../')
sys.path.append('../src/')

from time import time
from data_gen import *
from coreset import *
from utils_exp import *
from constants import DEFAULT_PARAMETERS
from result_averager import ResultAverager
from variational_trainer import VariationalTrainer

directory = "../exp/test"

dictUpdate = {
    'dataGen':SplitMnistGen(),
    'coresetMethod': coreset_rand,
    'numLayers' : (1,2),
    'coresetSize': 40,
    'numEpochs' : 120
}

for taskOrder in getAdversarialPermutationList():
    startTime = time()
    resultAverager = ResultAverager()
    dictUpdate['taskOrder'] = taskOrder
    dictParams = getAllExpParameters(dictUpdate)
    for iter in range(0,5):
        trainer = VariationalTrainer(dictParams)
        accuracy = trainer.train()
        resultAverager.add(accuracy)
    filename = getName(dictParams, 'taskOrder')
    writePerformanceRecordAccuracyAvg(directory, filename, resultAverager)
    print('Time for 1 task: {}'.format((time() - startTime) / 60))
    break
