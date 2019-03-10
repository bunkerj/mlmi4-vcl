import os
import sys
import pickle
import pathlib
import numpy as np
from copy import deepcopy
from statistics import stdev

AVERAGE = 'average'
STD = 'std'
PRINT = 'print'

def getResultsForAllExperiments(experimentName):
    results = {}
    folderDir = '{}/{}'.format(os.curdir, experimentName)
    for folderName in os.listdir(folderDir):
        targetDir = '{}/{}'.format(folderDir, folderName)
        batchOfExperiments = {}
        for filename in os.listdir(targetDir):
            targetFile = '{}/{}'.format(targetDir, filename)
            acc = pickle.load(open(targetFile, 'rb'))
            batchOfExperiments[filename] = acc
        results[folderName] = batchOfExperiments
    return results

def sumLists(list1, list2):
    if len(list1) != len(list2):
        raise Error('Length mismatch')
    list1Copy = deepcopy(list1)
    list2Copy = deepcopy(list2)
    for i in range(len(list1Copy)):
        if list1Copy[i] == None:
            continue
        list1Copy[i] += list2Copy[i]
    return list1Copy

def addToSum(sumOfResults, resultsList):
    for expName in resultsList:
        for task in range(len(resultsList[expName])):
            sumOfResults[expName][task] = \
                sumLists(sumOfResults[expName][task], resultsList[expName][task])
    return sumOfResults

def getAverage(sumOfResults, N):
    resultsList = deepcopy(sumOfResults)
    for expName in resultsList:
        for task in resultsList[expName]:
            for listIndex in range(len(resultsList[expName][task])):
                if resultsList[expName][task][listIndex] != None:
                    resultsList[expName][task][listIndex] /= N
    return resultsList

def getAverageResults(resultDict):
    sumOfResults = {}
    N = len(resultDict)
    for idx, experimentIdx in enumerate(resultDict):
        if idx == 0:
            sumOfResults = deepcopy(resultDict[experimentIdx])
        else:
            sumOfResults = addToSum(sumOfResults, resultDict[experimentIdx])
    return getAverage(sumOfResults, N)

def getStandardDeviation(resultDict):
    stdResults = {}
    if len(resultDict) == 0:
        raise Error('Results dictionary cannot be empty')
    firstExperimentIdx = list(resultDict)[0]
    for experimentName in resultDict[firstExperimentIdx]:
        stdResults[experimentName] = {}
        for taskIdx in range(len(resultDict[firstExperimentIdx][experimentName])):
            for idx, experimentIdx in enumerate(resultDict):
                newRow = resultDict[experimentIdx][experimentName][taskIdx]
                if idx == 0:
                    values = np.array(newRow, dtype=np.float64)
                values = np.vstack([values, np.array(newRow, dtype=np.float64)])
            values = np.std(values, 0)
            stdResults[experimentName][taskIdx] = list(values)
    return stdResults

def serializeEachExperiment(averageResults, experimentName):
    resultsDir = '{}/{}/{}'.format(os.curdir, 'averaged_results', experimentName)
    pathlib.Path(resultsDir).mkdir(parents=True, exist_ok=True)
    for expName in averageResults:
        expResultFilePath = '{}/{}'.format(resultsDir, expName)
        pickle.dump(averageResults[expName], open(expResultFilePath, "wb"))

def getStatisticalResult(statistic, resultDict):
    if statistic == AVERAGE:
        return getAverageResults(resultDict)
    elif statistic == STD:
        return getStandardDeviation(resultDict)
    else:
        raise Error('Invalid statistic: {}'.format(statistic))

def getAverageFromStatisticalResult(statisticalResult):
    averageResults = {}
    for experimentName in statisticalResult:
        count = 0
        averageResults[experimentName] = 0
        for taskIdx in statisticalResult[experimentName]:
            for value in statisticalResult[experimentName][taskIdx]:
                if not np.isnan(value):
                    count += 1
                    averageResults[experimentName] += value
        averageResults[experimentName] /= count
    return averageResults

def main():
    experimentName = sys.argv[1]
    statistic = sys.argv[2]
    action = sys.argv[3]

    resultDict = getResultsForAllExperiments(experimentName)
    statisticalResult = getStatisticalResult(statistic, resultDict)

    if action == PRINT:
        print(getAverageFromStatisticalResult(statisticalResult))
    else:
        serializeEachExperiment(statisticalResult, experimentName)

    print('\nProcess completed successfully.')

    # --------------------------------------------------------------- #
    # print(resultDict)
    # print(resultDict['exp_3_0'])
    # print(resultDict['exp_3_0']['SN_VCL.p'])
    # print(resultDict['exp_3_0']['SN_VCL.p'][0])

if __name__ == '__main__':
    main()
