import numpy as np
import torch
import torch.nn as nn

class KL():
    def __init__(self, qPos, qPri):
        pass

    def _getKL(self, m, v, m0, v0, layerId, parId):
        constTerm = ( - 0.5 * m.size()[0] * m.size()[1] if parId == WEIGHT
                        else -0.5 * outputSize)
        logStdDiff = 0.5 * torch.sum(np.log(v0) - v)
        muDiffTerm = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
        return constTerm + logStdDiff + muDiffTerm

    def klTerm(self):
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

        for taskId in range(qpos.getTaskCount()):
            for layerId, m in enumerate(qpos.getHead(WEIGHT, MEAN, taskId)):
                v = qpos.getHead(WEIGHT, VARIANCE, taskId)[layerId]
                m0 = qpri.getHead(WEIGHT, MEAN, taskId)[layerId]
                v0 = qpri.getHead(WEIGHT, VARIANCE, taskId)[layerId]
                kl += self._getKL(m, v, m0, v0, WEIGHT)

        for taskId in range(qpos.getTaskCount()):
            for layerId, m in enumerate(qpos.getHead(BIAS, MEAN, taskId)):
                v = qpos.getHead(BIAS, VARIANCE, taskId)[layerId]
                m0 = qpri.getHead(BIAS, MEAN, taskId)[layerId]
                v0 = qpri.getHead(BIAS, VARIANCE, taskId)[layerId]
                kl += self._getKL(m, v, m0, v0, WEIGHT)


        return kl
