
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class NeuralTrainer():
    def __init__(self, neuralNetwork):
        # Create random Tensors to hold input and output
        self.neuralNetwork = neuralNetwork

    def _assignOptimizer(self, learning_rate = 0.001):
        self.train_step = torch.optim.Adam(self.neuralNetwork.parameters(), lr = learning_rate)

    def train(self, x_train, y_train, task_id = 0, no_epochs=1000, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N
        # Training cycle
        costs = [];
        for epoch in range(no_epochs):
            perm_inds = list(range(x_train.shape[0]))
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            for i in range(total_batch):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                y_pred = self.neuralNetwork(batch_x)
                loss = self.neuralNetwork.loss(y_pred, batch_y)
                self._assignOptimizer()
                self.train_step.zero_grad()
                loss.backward()
                self.train_step.step()
                # Compute average loss
                avg_cost += loss / total_batch
                if (i+1) % display_epoch == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, no_epochs, i+1, total_batch, avg_cost))
            costs.append(avg_cost)
        print("Optimization Finished!")
        return costs


    def getWeights(self):
        weights = self.neuralNetwork.parameters()
        return weights
