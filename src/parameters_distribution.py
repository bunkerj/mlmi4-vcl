import sys
sys.path.append('../')

import torch
import torch.autograd as autograd
from scipy.stats import truncnorm
from constants import FloatTensor, MEAN, VARIANCE, WEIGHT, BIAS, INIT_VARIANCE

PARAMETER_TYPES = [WEIGHT, BIAS]
STATISTICS = [MEAN, VARIANCE]

class ParametersDistribution:
    def __init__(self, sharedWeightDim, headWeightDim, headCount):
        """ dimension: (# layers, input size, output size) """
        self.shared = {}
        self.heads = {}
        self.headCount = headCount
        self.sharedLayerCount = sharedWeightDim[0]
        self.headLayerCount = headWeightDim[0]
        for parameterType in PARAMETER_TYPES:
            self.shared[parameterType] = {}
            self.heads[parameterType] = {}
            for statistic in STATISTICS:
                self.shared[parameterType][statistic] = \
                    self.getParameters(sharedWeightDim, parameterType, False)
                self.heads[parameterType][statistic] = {}
                for head in range(headCount):
                    self.heads[parameterType][statistic][head] = \
                        self.getParameters(headWeightDim, parameterType, True)
        self.initializeMeansAndVariances(0, 1)

    def getCurrentWeightDimensions(self, layer, dimensions, isHead):
        layerCount, inputSize, outputSize = dimensions
        if isHead:
            currentInput = inputSize
            currentOutput = outputSize if layer == layerCount - 1 else inputSize
        else:
            currentInput = inputSize if layer == 0 else outputSize
            currentOutput = outputSize
        return currentInput, currentOutput

    def getCurrentBiasDimensions(self, layer, dimensions, isHead):
        layerCount, inputSize, outputSize = dimensions
        if isHead:
            currentOutput = outputSize if layer == layerCount - 1 else inputSize
        else:
            currentOutput = outputSize
        return currentOutput

    def getParameters(self, dimensions, parameterType, isHead):
        parameters = []
        for layer in range(dimensions[0]):
            if parameterType == WEIGHT:
                currentInput, currentOutput = self.getCurrentWeightDimensions(layer, dimensions, isHead)
                parameters.append(self.createTensor((currentInput, currentOutput)))
            elif parameterType == BIAS:
                currentOutput = self.getCurrentBiasDimensions(layer, dimensions, isHead)
                parameters.append(self.createTensor((currentOutput)))
        return parameters

    def _getTruncatedNormal(self, size, threshold=1):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        values = torch.from_numpy(values).type(FloatTensor)
        return values

    def getTruncatedNormalHeadMeans(self, headMeans, taskId):
        newHeadMeans = []
        for headMean in headMeans:
            newHeadMeans.append(self._getTruncatedNormal(headMean.size(), 0.02))
        return newHeadMeans

    def initializeHeads(self, taskId):
        for parameterType in PARAMETER_TYPES:
            self.heads[parameterType][MEAN][taskId] = self.getTruncatedNormalHeadMeans(self.heads[parameterType][MEAN][taskId], taskId)
            self.heads[parameterType][VARIANCE][taskId] = self.initializeTensorList(self.heads[parameterType][VARIANCE][taskId], INIT_VARIANCE)

    def initializeTensorList(self, tensorList, value):
        newTensorList = []
        for tensor in tensorList:
            newTensorList.append(value*torch.ones(tensor.size()).type(FloatTensor))
        return newTensorList

    def initializeMeansAndVariances(self, initMean, initVariance):
        for parameterType in PARAMETER_TYPES:
            self.shared[parameterType][MEAN] = self.initializeTensorList(self.shared[parameterType][MEAN], initMean)
            self.shared[parameterType][VARIANCE] = self.initializeTensorList(self.shared[parameterType][VARIANCE], initVariance)
            for head in range(self.headCount):
                self.heads[parameterType][MEAN][head] = self.initializeTensorList(self.heads[parameterType][MEAN][head], initMean)
                self.heads[parameterType][VARIANCE][head] = self.initializeTensorList(self.heads[parameterType][VARIANCE][head], initVariance)

    def createTensor(self, dimension):
        return autograd \
            .Variable(torch.rand(dimension) \
            .type(FloatTensor), requires_grad=True)

    def getShared(self, parameterType, statistic,  taskId = None): # Added taskId for convenience
        return self.shared[parameterType][statistic]

    def getHead(self, parameterType, statistic, head):
        return self.heads[parameterType][statistic][head]

    def getFlattenedParameters(self, head):
        return self.getShared(WEIGHT, MEAN) + \
        self.getShared(BIAS, MEAN) + \
        self.getHead(WEIGHT, MEAN, head) + \
        self.getHead(BIAS, MEAN, head)

    def getGenericListOfTensors(self, referenceList):
        tensorList = []
        for item in referenceList:
            variance = torch \
                .ones(item.size(), requires_grad=True) \
                .type(FloatTensor) * INIT_VARIANCE
            tensorList.append(variance)
        return tensorList

    def getInitialVariances(self, parameters):
        tensorList = self.getGenericListOfTensors(parameters)
        return (tensorList[:self.sharedLayerCount], \
                tensorList[self.sharedLayerCount:])

    def getInitialMeans(self, parameters):
        return (parameters[:self.sharedLayerCount], \
                parameters[self.sharedLayerCount:])

    def setParameters(self, parameters, taskId):
        weights, biases = parameters

        sharedWeightMeans, headWeightMeans = self.getInitialMeans(weights)
        sharedBiasMeans, headBiasMeans = self.getInitialMeans(biases)
        sharedWeightVariances, headWeightVariances = self.getInitialVariances(weights)
        sharedBiasVariances, headBiasVariances = self.getInitialVariances(biases)

        self.shared[WEIGHT][MEAN] = sharedWeightMeans
        self.shared[BIAS][MEAN] = sharedBiasMeans
        self.shared[WEIGHT][VARIANCE] = sharedWeightVariances
        self.shared[BIAS][VARIANCE] = sharedBiasVariances

        # for taskId in range(self.headCount):
        self.heads[WEIGHT][MEAN][taskId] = headWeightMeans
        self.heads[BIAS][MEAN][taskId] = headBiasMeans
        self.heads[WEIGHT][VARIANCE][taskId] = headWeightVariances
        self.heads[BIAS][VARIANCE][taskId] = headBiasVariances

    def setHeadParameters(self, parameters, taskId):
        weights, biases = parameters

        sharedWeightMeans, headWeightMeans = self.getInitialMeans(weights)
        sharedBiasMeans, headBiasMeans = self.getInitialMeans(biases)
        sharedWeightVariances, headWeightVariances = self.getInitialVariances(weights)
        sharedBiasVariances, headBiasVariances = self.getInitialVariances(biases)

        self.heads[WEIGHT][MEAN][taskId] = headWeightMeans
        self.heads[BIAS][MEAN][taskId] = headBiasMeans
        self.heads[WEIGHT][VARIANCE][taskId] = headWeightVariances
        self.heads[BIAS][VARIANCE][taskId] = headBiasVariances

    def purifyTensorList(self, tensorList):
        newTensorList = []
        for tensor in tensorList:
            tensor.detach_()
            newTensor = tensor.detach().clone()
            newTensor.requires_grad = True
            newTensorList.append(newTensor)
        return newTensorList

    def getSizesInTensorList(self, tensorList):
        return [t.size() for t in tensorList]

    def initializeHead(self, sourceHead, newHead):
        self.heads[WEIGHT][MEAN][newHead] = self.purifyTensorList(self.heads[WEIGHT][MEAN][sourceHead])
        self.heads[BIAS][MEAN][newHead] = self.purifyTensorList(self.heads[BIAS][MEAN][sourceHead])
        self.heads[WEIGHT][VARIANCE][newHead] = self.purifyTensorList(self.heads[WEIGHT][VARIANCE][sourceHead])
        self.heads[BIAS][VARIANCE][newHead] = self.purifyTensorList(self.heads[BIAS][VARIANCE][sourceHead])

    def printBodySum(self):
        parameters = self.getShared(BIAS, MEAN) + self.getShared(WEIGHT, MEAN)
        print(sum(p.sum() for p in parameters))

    def printHeadSum(self):
        for head in range(self.headCount):
            parameters = self.getHead(BIAS, MEAN, head) + self.getHead(WEIGHT, MEAN, head)
            print('Head: {} ---- Head sum: {}'.format(head, sum(p.sum() for p in parameters)))

    def printHeadDim(self, parameterType):
        for head in range(self.headCount):
            parameters = self.getHead(parameterType, MEAN, head)
            headDims = [p.size() for p in parameters]
            print('Head dims: {}'.format(headDims))

    def printSharedDim(self, parameterType):
        parameters = self.getShared(parameterType, MEAN)
        sharedDims = [p.size() for p in parameters]
        print('Shared dims: {}'.format(sharedDims))

    def printArchitectureDimensions(self):
        print('------ Distribution Architecture ------')
        print('*** WEIGHT ***')
        self.printHeadDim(WEIGHT)
        self.printSharedDim(WEIGHT)
        print('*** BIAS ***')
        self.printHeadDim(BIAS)
        self.printSharedDim(BIAS)
        print('---------------------------------------')

    def overwrite(self, q, onlyShared = False):
        for parameterType in PARAMETER_TYPES:
            for statistic in STATISTICS:
                self.shared[parameterType][statistic] = \
                    self.purifyTensorList(q.shared[parameterType][statistic])
                if not onlyShared:
                    for head in range(self.headCount):
                        self.heads[parameterType][statistic][head] = \
                            self.purifyTensorList(q.heads[parameterType][statistic][head])
