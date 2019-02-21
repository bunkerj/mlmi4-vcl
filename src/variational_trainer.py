from parameters_distribution import ParametersDistribution
from data_gen import *
from coreset import *
<<<<<<< Updated upstream
from optimizer import minimizeLoss

#sharedDim = (3, 3, 3)
#headDim = (2, 3, 1)
#headCount = 3
#sharedDim, headDim, headCount,

# Specify the experiment setting
# 1. Data generator - MnistGen() / SplitMnistGen() / PermutedMnistGen()
#                   - NotMnistGen() / SplitNotMnistGen() / PermutedNotMnistGen()
dataGen = SplitMnistGen()
# 2. Coreset method and size - coreset_rand / coreset_k
coresetMethod = coreset_rand
coresetSize = 20

hidden_size,
no_epochs,
data_gen,
coreset_method,
coreset_size=0,
batch_size=None,
single_head=True
=======
from copy import deepcopy
>>>>>>> Stashed changes

class VariationalTrainer:
    def __init__(self, numEpochs, batchSize, hiddenSize, numLayers, dataGen, coresetMethod, coresetSize):
        self.qPosterior = ParametersDistribution()

        self.numEpochs = numEpochs
        self.batchSize = batchSize

        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        self.dataGen = dataGen
        self.coresetMethod = coresetMethod
        self.coresetSize = coresetSize

    def train():
        # obtain the number of tasks
        num_tasks = self.dataGen.maxIter
        # initialize current iteration
        self.dataGen.curIter = 0
        # obtain input and output dimension
        input_dim, output_dim = self.dataGen.get_dims()
        # initialize coreset
        x_coresets = []
        y_coresets = []
        # initialize x_testsets
        x_testsets = []
        y_testsets = []

        for task_id in range(num_tasks):
            # train and test data for current task
            x_train, y_train, x_test, y_test = self.dataGen.next_task()
            # append test data to test sets
            x_testsets.append(x_test)
            y_testsets.append(y_test)

            # initialize the network with maximum likelihood weights
            if task_id == 0:
                model = VanillaNN(input_dim, self.hiddenSize, self.numLayers, output_dim)
                modelTrainer = NeuralTrainer(model)
                modelTrainer.train(x_train, y_train, self.noEpochs, self.batchSize, displayEpoch = 20)
                param_mean = model.getParameters()
                # use parameter mean to initialize the q posterior
                self.qPosterior.setParameters(param_mean)

            # create coreset
            if self.coresetSize > 0:
                x_coresets, y_coresets, x_train, y_train = self.coresetMethod(x_coresets, y_coresets, x_train, y_train, self.coresetSize)

            # update weights and bias for current task
            self.qPosterior.overwrite(maximizeVariationalLowerBound(qPosterior, x_train, y_train))

            # incorporate coreset data and make prediction
            qPred = deepcopy(self.qPosterior)
            qPred.overwrite(maximizeVariationalLowerBound(qPred, x_))







def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y

def get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size=None):
    mf_weights, mf_variances = model.get_weights()
    acc = []

    if single_head:
        if len(x_coresets) > 0:
            x_train, y_train = merge_coresets(x_coresets, y_coresets)
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            final_model = MFVI_NN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
            final_model.train(x_train, y_train, 0, no_epochs, bsize)
        else:
            final_model = model

    for i in range(len(x_testsets)):
        if not single_head:
            if len(x_coresets) > 0:
                x_train, y_train = x_coresets[i], y_coresets[i]
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = MFVI_NN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
                final_model.train(x_train, y_train, i, no_epochs, bsize)
            else:
                final_model = model

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]

        pred = final_model.prediction_prob(x_test, head)
        pred_mean = np.mean(pred, axis=0)
        pred_y = np.argmax(pred_mean, axis=1)
        # I got lost here (could anyone explain this to me?)
        y = np.argmax(y_test, axis=1)
        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)

        if len(x_coresets) > 0 and not single_head:
            final_model.close_session()

    if len(x_coresets) > 0 and single_head:
        final_model.close_session()

    def self.getBatch(x_train, y_train):
        self.batchSize
        batches = []
        numberOfBatches = x_train.size() / self.batchSize
        if isinstance(numberOfBatches, int):
            errMessage = 'Batch size {} not consistent with dataset size {}' \
                .format(x_train.size(), self.batchSize)
            raise Exception(errMessage)
        for i in range(numberOfBatches):
            start = i*self.batchSize
            end = (i+1)*self.batchSize
            x_train_batch = x_train[start:end]
            y_train_batch = y_train[start:end]
            batches.append((x_train_batch, y_train_batch))
        return batches

    def maximizeVariationalLowerBound(x_train_batch, y_train_batch, qPrior, taskId):
        qPosterior = ParametersDistribution()
        parameters = qPosterior.getFlattenedParameters(taskId)
        optimizer = torch.optim.Adam(parameters, lr = 0.001)
        for x_train_batch, y_train_batch in self.getBatch(x_train, y_train):
            lossArgs = (x_train_batch, y_train_batch qPosterior, qPrior, taskId)
            minimizeLoss(1000, optimizer, getCost, lossArgs)
        return qPosterior
