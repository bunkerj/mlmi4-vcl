import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from constants import MEAN, VARIANCE, WEIGHT, BIAS

class KL:
    def _getKL(self, m, v, m0, v0, parId):
        constTerm = -0.5 * m.numel()
        logStdDiff = 0.5 * torch.sum(v0 - v)
        muDiffTerm = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / torch.exp(v0))
        return constTerm + logStdDiff + muDiffTerm

    def computeKL(self, qPos, qPri, taskId):
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

        for taskId_ in range(taskId):
            for layerId, m in enumerate(qPos.getHead(WEIGHT, MEAN, taskId_)):
                v = qPos.getHead(WEIGHT, VARIANCE, taskId_)[layerId]
                m0 = qPri.getHead(WEIGHT, MEAN, taskId_)[layerId]
                v0 = qPri.getHead(WEIGHT, VARIANCE, taskId_)[layerId]
                kl += self._getKL(m, v, m0, v0, WEIGHT)

            for layerId, m in enumerate(qPos.getHead(BIAS, MEAN, taskId_)):
                v = qPos.getHead(BIAS, VARIANCE, taskId_)[layerId]
                m0 = qPri.getHead(BIAS, MEAN, taskId_)[layerId]
                v0 = qPri.getHead(BIAS, VARIANCE, taskId_)[layerId]
                kl += self._getKL(m, v, m0, v0, BIAS)

        return kl
