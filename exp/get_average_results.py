import os
import sys
import pickle
import pathlib
from copy import deepcopy

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
    for idx, key in enumerate(resultDict):
        if idx == 0:
            sumOfResults = deepcopy(resultDict[key])
        else:
            sumOfResults = addToSum(sumOfResults, resultDict[key])
    return getAverage(sumOfResults, N)

def serializeEachExperiment(averageResults, experimentName):
    resultsDir = '{}/{}/{}'.format(os.curdir, 'results', experimentName)
    pathlib.Path(resultsDir).mkdir(parents=True, exist_ok=True)
    for expName in averageResults:
        expResultFilePath = '{}/{}'.format(resultsDir, expName)
        pickle.dump(averageResults[expName], open(expResultFilePath, "wb"))

def main():
    experimentName = sys.argv[1]
    resultDict = getResultsForAllExperiments(experimentName)
    averageResults = getAverageResults(resultDict)
    serializeEachExperiment(averageResults, experimentName)
    print('Done :)')

if __name__ == '__main__':
    main()
