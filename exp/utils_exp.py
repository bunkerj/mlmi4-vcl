import sys
sys.path.append('../')
sys.path.append('../src/')
import itertools
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
