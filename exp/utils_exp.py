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
        print("Value : %s" %  updatedDictionary)

    return updatedDictionary


def getAdversarialOrderList():

    pass
