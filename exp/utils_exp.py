import sys
sys.path.append('../')
sys.path.append('../src/')
import itertools
import pickle
import pathlib
from copy import deepcopy
from constants import DEFAULT_PARAMETERS

def getAllExpParameters(updateObj, visUpdate = True):

    updatedDictionary  = deepcopy(DEFAULT_PARAMETERS)
    updatedDictionary.update(updateObj)
    if visUpdate:
        print("Experiments Settings : %s" %  updatedDictionary)

    return updatedDictionary

def getAdversarialPermutationList():

    baselineOrder = [1,2,3,4]
    allPermutations = [list(item) for item in itertools.permutations(baselineOrder)]
    allPermutations = [[0] + item for item in allPermutations]
    return allPermutations

def getName(dictionary, dictEntry):
    return dictEntry+'_'+str(dictionary[dictEntry]).replace('[', '').replace(']', '').replace(', ', '_')

def getPath(directory, dictionary, dictEntry):
    filename = getName(dictionary, dictEntry)
    return '{}/{}'.format(directory, filename)

def writeToFile(obj, path):
    pathlib.Path(resultsDir).mkdir(parents=True, exist_ok=True)
    pickle.dump(obj, open(filename, "wb"))

def writePerformanceRecordAccuracyAvg(path, resultAverager):
    averagePR = resultAverager.getAveragePerformanceRecord()
    averagePRaverage = resultAverager.getAveragePerformanceRecordAverage()
    pathWithAverageSuffix = '{}_average_{}'.format(path, averagePRaverage)
    writeToFile(accuracy, pathWithAverageSuffix)