import numpy as np
import torch
import torch.nn as nn

class KL():
    def __init__(self, qPos, qPri):
        super(cost, self).__init__(neuralNetwork)
        self.qPos = qPos
        self.qPri = qPri
        self.wPos = self.qPos.weight
        self.bPos = self.qPos.bias
        self.wPri = self.qPri.weight
        self.bPri = self.qPri.bias

    def KLTerm(self):
        kl = 0
        for layerId in range(self.numLayersShell):
            outputSize, inputSize = self.getLayerDimensions(i)
            constTerm = [- 0.5 * outputSize * inputSize, -0.5 * dout]
            # computing the weight term
            m, v = self.wPos.mean[layerId], self.wPos.sigma[layerId]
            m0, v0 = self.wPri.mean[layerId], self.wPri.sigma[layerId]
            logStdDiff = 0.5 * torch.sum(np.log(v0) - v)
            muDiffTerm = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += constTerm[0] + logStdDiff + muDiffTerm
            # computing the bias term
            m, v = self.bPos.mean[layerId], self.bPos.sigma[layerId]
            m0, v0 = self.bPri.mean[layerId], self.bPri.sigma[layerId]
            logStdDiff = 0.5 * torch.sum(np.log(v0) - v)
            muDiffTerm = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += constTerm[1] + logStdDiff + muDiffTerm


        for layerId in range(self.numLayersHeads):
            outputSize, inputSize = self.getLayerDimensions(layerId + self.numLayersShell)
            constTerm = [- 0.5 * outputSize * inputSize, -0.5 * dout]
            noTasks = len(self.wPos.mean[layerId])
            for taskId in range(noTasks):
                # computing the weight term
                m, v = self.wPos.mean[layerId][taskId], self.wPos.sigma[layerId][taskId]
                m0, v0 = self.wPri.mean[layerId][taskId], self.wPri.sigma[layerId][taskId]
                logStdDiff = 0.5 * torch.sum(np.log(v0) - v)
                muDiffTerm = 0.5 * torch.sum((tf.exp(v) + (m0 - m)**2) / v0)
                kl += constTerm[0] + logStdDiff + muDiffTerm
                # computing the bias term
                m, v = self.bPos.mean[layerId][taskId], self.bPos.sigma[layerId][taskId]
                m0, v0 = self.bPri.mean[layerId][taskId], self.bPri.sigma[layerId][taskId]
                logStdDiff = 0.5 * torch.sum(np.log(v0) - v)
                muDiffTerm = 0.5 * torch.sum((tf.exp(v) + (m0 - m)**2) / v0)
                kl += constTerm[1] + logStdDiff + muDiffTerm

        return kl
