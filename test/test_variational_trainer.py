import sys
sys.path.append('../')
sys.path.append('../src/')

from data_gen import *
from coreset import *
from variational_trainer import VariationalTrainer

# default setup
dictParams = {
'numEpochs':120,
'alpha':1,
'dataGen':SplitMnistGen(set0 = [0,2,4,6,8], set1 = [1,3,5,7,9]),
'numTasks':5,
'numHeads':5,
'numLayers':(2,1),
'hiddenSize':256,
'taskOrder':[0,1,2,3,4],
'headOrder':[0,1,2,3,4],
}

# Pickle file naming convention:
# {0}_{1}_{2}_{3}.p
# {0}: data gen (e.g. Permuted Mnist -> PM, Split Mnist -> SM, Permuted NotMnist = PN, Split NotMnist = SN)
# {1}: method (e.g. VCL)
# {2}: coreset (e.g. Random Coreset -> RC, Random Coreset Only -> RCO)
# {3}: coreset size

# 1. VCL (no Coreset)
dictParams['batchSize'] = None
dictParams['coresetSize'] = 0
dictParams['coresetMethod'] = coreset_rand
dictParams['coresetOnly'] = False

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()

# run experiment
print(trainer.accuracy)
