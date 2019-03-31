import sys
sys.path.append('../')
sys.path.append('../src/')

from data_gen import *
from coreset import *
from utils_exp import *
from constants import DEFAULT_PARAMETERS
from result_averager import ResultAverager
from variational_trainer import VariationalTrainer

dictUpdate = {'dataGen':SplitMnistGen(),'coresetMethod': coreset_rand,'numLayers' : (1,2), 'coresetSize': 40}
dictParams = getAllExpParameters(dictUpdate)
for taskOrder in getAdversarialPermutationList():
    resultAverager = ResultAverager()
    dictUpdate = {'taskOrder' : taskOrder}
    dictParams = getAllExpParameters(dictUpdate)
    for iter in range(0,5):
        trainer = VariationalTrainer(dictParams)
        accuracy = trainer.train()
        resultAverager.add(accuracy)
        path = getPath(dictionary, dictEntry)
        writePerformanceRecordAccuracyAvg(path, resultAverager)
