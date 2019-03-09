################################################################################
# Experiment 1. Permuted Mnist - Learning Methods & Coreset Size
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
'numEpochs' : 100,
'alpha' : 1,
'dataGen' : PermutedMnistGen(),
'numTasks' : 10,
'numHeads' : 1,
'numLayers' : (2,1),
'hiddenSize' : 100,
'taskOrder' : [],
'headOrder' : [],
}

# Pickle file naming convention:
# {0}_{1}_{2}_{3}.p
# {0}: data gen (e.g. Permuted Mnist -> PM, Split Mnist -> SM, Permuted NotMnist = PN, Split NotMnist = SN)
# {1}: method (e.g. VCL)
# {2}: coreset (e.g. Random Coreset -> RC, Random Coreset Only -> RCO)
# {3}: coreset size

# 1. VCL (no Coreset)
dictParams['batchSize'] = 256
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 0
dictParams['coresetMethod'] = None

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_1/PM_VCL.p"
pickle.dump(accuracy, open(filename, "wb"))

# 2 . VCL + Random Coreset
dictParams['batchSize'] = 256
dictParams['coresetOnly'] = False
dictParams['coresetMethod'] = coreset_rand

for size in [200, 400, 1000, 2500, 5000]:
    dictParams['coresetSize'] = size
    trainer = VariationalTrainer(dictParams)
    accuracy = trainer.train()
    filename = "exp_1/PM_VCL_RC_{}.p".format(size)
    pickle.dump(accuracy, open(filename, "wb"))

# 3. Random Coreset only
dictParams['batchSize'] = 200
dictParams['coresetOnly'] = True
dictParams['coresetMethod'] = coreset_rand

for size in [200, 400, 1000, 2500, 5000]:
    dictParams['coresetSize'] = size
    trainer = VariationalTrainer(dictParams)
    accuracy = trainer.train()
    filename = "exp_1/PM_VCL_RCO_{}.p".format(size)
    pickle.dump(accuracy, open(filename, "wb"))

# 4. VCL + K-center Coreset
dictParams['batchSize'] = 256
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 200
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_1/PM_VCL_KC_200.p"
pickle.dump(accuracy, open(filename, "wb"))

# 5. K-center Coreset only
dictParams['batchSize'] = 200
dictParams['coresetOnly'] = True
dictParams['coresetSize'] = 200
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_1/PM_VCL_KCO_200.p"
pickle.dump(accuracy, open(filename, "wb"))
