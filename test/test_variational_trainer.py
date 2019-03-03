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
'numSamples':100,
'dataGen':PermutedMnistGen(),
'numTasks':2,
'numHeads':1,
'coresetMethod':coreset_k,
'coresetSize':200,
'numLayers':(2,1),
'hiddenSize':256,
'taskOrder':[],
'headOrder':[],
}

# run experiment
trainer = VariationalTrainer(dictParams)
trainer.train()
print(trainer.accuracy)
