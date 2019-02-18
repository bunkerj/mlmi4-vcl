
import numpy as np
import torch
import torch.nn as nn


class NeuralTrainer():
    def __init__(self, neuralNetwork):
        # Create random Tensors to hold input and output
        self.neuralNetwork = neuralNetwork

    def _assignOptimizer(self, learningRate = 0.001):
        self.train_step = torch.optim.Adam(self.neuralNetwork.parameters(), lr = learningRate)

    def train(self, xTrain, yTrain, taskId = 0, noEpochs=1000, batchSize=100, displayEpoch=5):
        N = xTrain.shape[0]
        if batchSize > N:
            batchSize = N
        # Training cycle
        costs = [];
        for epoch in range(noEpochs):
            permInds = list(range(xTrain.shape[0]))
            np.random.shuffle(permInds)
            curxTrain = xTrain[permInds]
            curyTrain = yTrain[permInds]

            avgCost = 0.
            totalBatch = int(np.ceil(N * 1.0 / batchSize))
            for i in range(totalBatch):
                startInd = i*batchSize
                endInd = np.min([(i+1)*batchSize, N])
                xBatch = curxTrain[startInd:endInd, :]
                yBatch = curyTrain[startInd:endInd, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                yPred = self.neuralNetwork(xBatch)
                loss = self.neuralNetwork.loss(yPred, yBatchS)
                self._assignOptimizer()
                self.train_step.zero_grad()
                loss.backward()
                self.train_step.step()
                # Compute average loss
                avgCost += loss / totalBatch
                if (i+1) % displayEpoch == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, noEpochs, i+1, totalBatch, avgCost))
            costs.append(avgCost)
        print("Optimization Finished!")
        return costs
