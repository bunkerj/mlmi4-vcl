################################################################################
# Experiment 2. Split Mnist - Learning Methods & Coreset Size
################################################################################

import sys
sys.path.append('../')
sys.path.append('../src/')

from data_gen import *
from coreset import *
from variational_trainer import VariationalTrainer
import pickle

# default setup
dictParams = {
'numEpochs' : 120,
'batchSize' : None,
'alpha' : 1,
'dataGen' : SplitMnistGen(),
'numTasks' : 5,
'numHeads' : 5,
'numLayers' : (2,1),
'hiddenSize' : 256,
'taskOrder' : [],
'headOrder' : [],
'coresetOnly': False,
'coresetSize': 0
}

# Pickle file naming convention:
# {0}_{1}_{2}_{3}.p
# {0}: data gen (e.g. Permuted Mnist -> PM, Split Mnist -> SM, Permuted NotMnist = PN, Split NotMnist = SN)
# {1}: method (e.g. VCL)
# {2}: coreset (e.g. Random Coreset -> RC, Random Coreset Only -> RCO)
# {3}: coreset size

1. VCL (no Coreset)
for i in range(5):
    dictParams['numLayers'] = (1,2)
    trainer = VariationalTrainer(dictParams)
    accuracy = trainer.train()
    filename = "exp_6/1_2_iter_{}.p".format(i)
    pickle.dump(accuracy, open(filename, "wb"))

    # 2 . VCL + Random Coreset
    dictParams['numLayers'] = (1,3)
    trainer = VariationalTrainer(dictParams)
    accuracy = trainer.train()
    filename = "exp_6/1_3_iter_{}.p".format(i)
    pickle.dump(accuracy, open(filename, "wb"))

    # 3. VCL + K-center Coreset
    dictParams['numLayers'] = (1,4)
    trainer = VariationalTrainer(dictParams)
    accuracy = trainer.train()
    filename = "exp_6/1_4_iter_{}.p".format(i)
    pickle.dump(accuracy, open(filename, "wb"))
