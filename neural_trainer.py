
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class NeuralTrainer():
    def __init__(self, neuralNetwork):
        # Create random Tensors to hold input and output
        self.neuralNetwork = neuralNetwork

    def _assignOptimizer(self, learning_rate = 0.001):
        self.train_step = torch.optim.Adam(model.parameters(), lr = learning_rate)

    def train(self, x_train, y_train, no_epochs=1000, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N
        # training cycle
        cost = [];
        for epoch in range(no_epochs):
            perm_inds = range(x_train.shape[0])
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
                self.x = batch_x
                self.y = batch_y
                y_pred = self.neuralNetwork.model(x) # MODEL NOT DEFINED YET
                loss = self.neuralNetwork.loss(y_pred, y) # THIS IS NOT DEFINED YET
                train_step.zero_grad()
                loss.backward()
                train_step.step()
                c = loss.data.item()
                # Compute average loss
                avg_cost += c / total_bat
            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            costs.append(avg_cost)
        print("Optimization Finished!")
        return costs

    def getWeights(self):
        weights = self.neuralNetwork.model.parameters()
        return weights
