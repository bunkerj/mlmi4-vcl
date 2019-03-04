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
'numEpochs':100,
'batchSize':256,
'numSamples':100,
'dataGen':PermutedMnistGen(),
'numTasks':10,
'numHeads':1,
'coresetMethod':None,
'coresetSize':0,
'coresetOnly':False,
'numLayers':(2,1),
'hiddenSize':100,
'taskOrder':[],
'headOrder':[],
}

# Pickle file naming convention:
# {0}_{1}_{2}_{3}.p
# {0}: data gen (e.g. Permuted Mnist -> PM, Split Mnist -> SM, Permuted NotMnist = PN, Split NotMnist = SN)
# {1}: method (e.g. VCL)
# {2}: coreset (e.g. Random Coreset -> RC, Random Coreset Only -> RCO)
# {3}: coreset size

# 1. VCL (no Coreset)
trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/PM_VCL.p", "wb"))

# 2. VCL + K-center Coreset
dictParams['coresetSize'] = 200
dictParams['coresetMethod'] = coreset_k
trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/PM_VCL_KC_200.p", "wb"))

# 3. K-center Coreset only
dictParams['coresetOnly'] = True
dictParams['batchSize'] = 200
trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/PM_VCL_KCO_200.p", "wb"))

# 4. VCL + Random Coreset
dictParams['coresetOnly'] = False
dictParams['batchSize'] = 256
dictParams['coresetMethod'] = coreset_rand
for size in [200, 400, 1000, 2500, 5000]:
    dictParams['coresetSize'] = size
    trainer = VariationalTrainer(dictParams)
    accuracy = trainer.train()
    pickle.dump(accuracy, open( "results/PM_VCL_RC_{}.p".format(size), "wb"))

# 5. Random Coreset only
dictParams['coresetOnly'] = True
dictParams['batchSize'] = 200
for size in [200, 400, 1000, 2500, 5000]:
    dictParams['coresetSize'] = size
    trainer = VariationalTrainer(dictParams)
    accuracy = trainer.train()
    pickle.dump(accuracy, open( "results/PM_VCL_RCO_{}.p".format(size), "wb"))
