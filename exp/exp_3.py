################################################################################
# Experiment 2. Split Not Mnist - Learning Methods & Coreset Size
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
'dataGen' : SplitNotMnistGen(),
'numTasks' : 5,
'numHeads' : 5,
'numLayers' : (4,1),
'hiddenSize' : 150,
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
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 0
dictParams['coresetMethod'] = None

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SN_VCL.p", "wb"))

# 2 . VCL + Random Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_rand

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SN_VCL_RC_40.p", "wb"))

# 3. Random Coreset only
dictParams['coresetOnly'] = True
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_rand

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SN_VCL_RCO_40.p", "wb"))

# 4. VCL + K-center Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SN_VCL_KC_40.p", "wb"))

# 5. K-center Coreset only
dictParams['coresetOnly'] = True
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
pickle.dump(accuracy, open( "results/SN_VCL_KCO_40.p", "wb"))
