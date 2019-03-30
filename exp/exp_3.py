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
filename = "exp_2/SM_VCL.p"
pickle.dump(accuracy, open(filename, "wb"))

# 2 . VCL + Random Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_rand

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_RC_40.p"
pickle.dump(accuracy, open(filename, "wb"))

# 3. VCL + K-center Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 40
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_KC_40.p"
pickle.dump(accuracy, open(filename, "wb"))

# 4 . VCL + Random Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 80
dictParams['coresetMethod'] = coreset_rand

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_RC_80.p"
pickle.dump(accuracy, open(filename, "wb"))

# 5. VCL + K-center Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 80
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_KC_80.p"
pickle.dump(accuracy, open(filename, "wb"))

# 6 . VCL + Random Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 160
dictParams['coresetMethod'] = coreset_rand

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_RC_160.p"
pickle.dump(accuracy, open(filename, "wb"))

# 7. VCL + K-center Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 160
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_KC_160.p"
pickle.dump(accuracy, open(filename, "wb"))

# 8 . VCL + Random Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 320
dictParams['coresetMethod'] = coreset_rand

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_RC_320.p"
pickle.dump(accuracy, open(filename, "wb"))

# 9. VCL + K-center Coreset
dictParams['coresetOnly'] = False
dictParams['coresetSize'] = 320
dictParams['coresetMethod'] = coreset_k

trainer = VariationalTrainer(dictParams)
accuracy = trainer.train()
filename = "exp_2/SM_VCL_KC_320.p"
pickle.dump(accuracy, open(filename, "wb"))
