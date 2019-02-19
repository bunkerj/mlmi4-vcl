import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import MEAN, VARIANCE, WEIGHT, BIAS

class KL:
    def _getKL(self, m, v, m0, v0, parId):
        constTerm = ( - 0.5 * m.size()[0] * m.size()[1] if parId == WEIGHT
                        else -0.5 * m.size()[0])
        logStdDiff = torch.sum(torch.log(v0) - torch.log(v))
        muDiffTerm = 0.5 * torch.sum((v + (m0 - m)**2) / v0)
        return constTerm + logStdDiff + muDiffTerm

    def computeKL(self, taskId, qPos, qPri):
        kl = 0
        for layerId, m in enumerate(qPos.getShared(WEIGHT,MEAN)):
            v = qPos.getShared(WEIGHT, VARIANCE)[layerId]
            m0 = qPri.getShared(WEIGHT, MEAN)[layerId]
            v0 = qPri.getShared(WEIGHT, VARIANCE)[layerId]
            kl += self._getKL(m, v, m0, v0, WEIGHT)

        for layerId, m in enumerate(qPos.getShared(BIAS,MEAN)):
            v = qPos.getShared(BIAS, VARIANCE)[layerId]
            m0 = qPri.getShared(BIAS, MEAN)[layerId]
            v0 = qPri.getShared(BIAS, VARIANCE)[layerId]
            kl += self._getKL(m, v, m0, v0, BIAS)

        for layerId, m in enumerate(qPos.getHead(WEIGHT, MEAN, taskId)):
            v = qPos.getHead(WEIGHT, VARIANCE, taskId)[layerId]
            m0 = qPri.getHead(WEIGHT, MEAN, taskId)[layerId]
            v0 = qPri.getHead(WEIGHT, VARIANCE, taskId)[layerId]
            kl += self._getKL(m, v, m0, v0, WEIGHT)

        for layerId, m in enumerate(qPos.getHead(BIAS, MEAN, taskId)):
            v = qPos.getHead(BIAS, VARIANCE, taskId)[layerId]
            m0 = qPri.getHead(BIAS, MEAN, taskId)[layerId]
            v0 = qPri.getHead(BIAS, VARIANCE, taskId)[layerId]
            kl += self._getKL(m, v, m0, v0, BIAS)

        return kl
