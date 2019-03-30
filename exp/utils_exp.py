import sys
sys.path.append('../')
sys.path.append('../src/')

from data_gen import *
from coreset import *
from copy import deepcopy
from constants import DEFAULT_PARAMETERS

def getAllExpParameters(updateObj, visUpdate = True):

    updatedDictionary  = DEFAULT_PARAMETERS.deepcopy()
    updatedDictionary.update(updateObj)
    if visUpdate:
        print("Experiments Settings : %s" %  updatedDictionary)

    return updatedDictionary

def getAdversarialPermutationList():

    baselineOrder = [1,2,3,4]
    allPermutations = [list(item) for item in itertools.permutations(baselineOrder)]
    allPermutations = [[0] + item for item in allPermutations]
    pass task
