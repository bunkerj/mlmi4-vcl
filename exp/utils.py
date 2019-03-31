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

def writeToFile(obj, directory, filename):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    fullPath = '{}/{}'.format(directory, filename)
    pickle.dump(obj, open(fullPath, "wb"))

def writePerformanceRecordAccuracyAvg(directory, filename, resultAverager):
    averagePR = resultAverager.getAveragePerformanceRecord()
    averagePRaverage = resultAverager.getAveragePerformanceRecordAverage()
    filenameWithAverageSuffix = '{}_average_{}'.format(filename, averagePRaverage)
    writeToFile(averagePR, directory, filenameWithAverageSuffix)

def convert(n, base):
    T = "0123456789ABCDEF"
    q, r = divmod(n, base)
    if q == 0:
        return T[r]
    else:
        return convert(q, base) + T[r]

def getHeadOrderList(num_heads, num_tasks):

    head_order_list = []
    for i in range(num_heads ** num_tasks):

        head_order = list('%0{}d'.format(num_tasks) % int(convert(i, num_heads)))

        if len(set(head_order)) == num_heads:
            head_order_list.append(head_order)

    head_order_list2 = []
    for head_order in head_order_list:
        head_mapping = {}

        idx = 0
        for head in head_order:
            if head not in head_mapping.keys():
                head_mapping[head] = idx
                idx += 1

        for j in range(len(head_order)):
            head_order[j] = head_mapping[head_order[j]]

        if head_order not in head_order_list2:
            head_order_list2.append(head_order)

    return head_order_list2
