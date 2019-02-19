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

    def _getMeanAndVariance(self, layerId, parId, taskId):

        posPar, priPar = ( self.wPos[taskId], self.wPri[taskId] if taskId is not None
                            else self.wPos, self.wPri)
        parId = (0 if parId == "weight"
                 else 1)
        m, v = posPar[layerId][parId]['mean'], posPar[layerId][parId]['sigma']
        m0, v0 = priPar['mean'][layerId], priPar['sigma'][layerId]
        return {'m':m, 'v':v ,'m0':mo, 'v0': v0}


    def _getklUpdate(self, layerId, parId, taskId):

        outputSize, inputSize = self.getLayerDimensions(layerId)
        constTerm = ( - 0.5 * outputSize * inputSize if parId == "weight"
                        else -0.5 * dout)
        par = _getMeanAndVariance(layerId, parId, taskId)
        logStdDiff = 0.5 * torch.sum(np.log(par['v0']) - par['v'])
        muDiffTerm = 0.5 * torch.sum((torch.exp(par['v']) + (par['m0']) - par['m']))**2) / par['v0']))
        return constTerm + logStdDiff + muDiffTerm


    def klTerm(self):
        kl = 0
        for layerId in range(self.numLayersShell):
            # computing the weight term
            kl += self._getklUpdate(layerId,"weight", taskId = None)
            # computing the bias term
            kl += self._getklUpdate(layerId,"bias", taskId = None)

        for layerId in range(self.numLayersHeads):
            noTasks = len(_getMeanAndVariance(layerId + self.numLayersShell, "weight", taskId = None)['m'])
            for taskId in range(noTasks):
                # computing the weight term
                kl += self._getklUpdate(layerId + self.numLayersShell, "weight", taskId)
                # computing the bias term
                kl += self._getklUpdate(layerId + self.numLayersShell, "bias", taskId)

        return kl
