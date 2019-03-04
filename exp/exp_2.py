################################################################################
# Experiment 2. Split Mnist - Learning Methods
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
'numEpochs':120,
'numSamples':100,
'dataGen':SplitMnistGen(),
'numTasks':5,
'numHeads':5,
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
dictParams['batchSize'] = None #same as training set
dictParams['coresetSize'] = 0
dictParams['coresetMethod'] = None
dictParams['coresetOnly'] = False

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SM_VCL.p", "wb"))

# 2. VCL + K-center Coreset
dictParams['batchSize'] = None #same as training set
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_k
dictParams['coresetOnly'] = False

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SM_VCL_KC_40.p", "wb"))

# 3. K-center Coreset only
dictParams['batchSize'] = 120
dictParams['coresetSize'] = 120
dictParams['coresetMethod'] = coreset_k
dictParams['coresetOnly'] = True

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SM_VCL_KCO_120.p", "wb"))

# 4. VCL + Random Coreset
dictParams['batchSize'] = None #same as training set
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_rand
dictParams['coresetOnly'] = False

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/PM_VCL_RC_40.p".format(size), "wb"))

# 5. Random Coreset only
dictParams['batchSize'] = 120
dictParams['coresetSize'] = 120
dictParams['coresetMethod'] = coreset_rand
dictParams['coresetOnly'] = True

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/PM_VCL_RCO_120.p".format(size), "wb"))
