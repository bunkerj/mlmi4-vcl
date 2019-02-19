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

    def computeKL(self, taskId, qpos, qpri):
        kl = 0
        for layerId, m in enumerate(qpos.getShared(WEIGHT,MEAN)):
            v = qpos.getShared(WEIGHT, VARIANCE)[layerId]
            m0 = qpri.getShared(WEIGHT, MEAN)[layerId]
            v0 = qpri.getShared(WEIGHT, VARIANCE)[layerId]
            kl += self._getKL(m, v, m0, v0, WEIGHT)

        for layerId, m in enumerate(qpos.getShared(BIAS,MEAN)):
            v = qpos.getShared(BIAS, VARIANCE)[layerId]
            m0 = qpri.getShared(BIAS, MEAN)[layerId]
            v0 = qpri.getShared(BIAS, VARIANCE)[layerId]
            kl += self._getKL(m, v, m0, v0, BIAS)

        for layerId, m in enumerate(qpos.getHead(WEIGHT, MEAN, taskId)):
            v = qpos.getHead(WEIGHT, VARIANCE, taskId)[layerId]
            m0 = qpri.getHead(WEIGHT, MEAN, taskId)[layerId]
            v0 = qpri.getHead(WEIGHT, VARIANCE, taskId)[layerId]
            kl += self._getKL(m, v, m0, v0, WEIGHT)

        for layerId, m in enumerate(qpos.getHead(BIAS, MEAN, taskId)):
            v = qpos.getHead(BIAS, VARIANCE, taskId)[layerId]
            m0 = qpri.getHead(BIAS, MEAN, taskId)[layerId]
            v0 = qpri.getHead(BIAS, VARIANCE, taskId)[layerId]
            kl += self._getKL(m, v, m0, v0, BIAS)

        return kl
