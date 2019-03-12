import sys
sys.path.append('../')
sys.path.append('../src/')

from data_gen import *
from coreset import *
from variational_trainer import VariationalTrainer

# experiment setup
dictParams = {
'numEpochs':1,
'batchSize':100,
'numSamples':10000,
'dataGen':PermutedMnistGen(),
'numTasks':5,
'numHeads':5,
'coresetMethod':coreset_rand,
'coresetSize':0,
'numLayers':(2,1),
'hiddenSize':256,
'taskOrder':[],
'headOrder':[],
}

# run experiment
trainer = VariationalTrainer(dictParams)
trainer.train()
print(trainer.accuracy)
