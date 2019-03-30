class ResultAverager:
    def __init__(self):
        self.resultCount = 0
        self.performanceRecordSum = None
        self.didPerformAveraging = False

    def add(self, performanceRecord):
        if self.didPerformAveraging:
            raise Exception('Cannot add to previously averaged results')
        elif self.performanceRecordSum == None:
            self.performanceRecordSum = performanceRecord
        else:
            for taskKey in performanceRecord:
                for idx in range(len(performanceRecord[taskKey])):
                    if performanceRecord[taskKey][idx] != None:
                        self.performanceRecordSum[taskKey][idx] += performanceRecord[taskKey][idx]
        self.resultCount += 1

    def getAveragePerformanceRecord(self):
        for taskKey in self.performanceRecordSum:
            for idx in range(len(self.performanceRecordSum[taskKey])):
                if self.performanceRecordSum[taskKey][idx] != None:
                    self.performanceRecordSum[taskKey][idx] /= self.resultCount
        self.didPerformAveraging = True
        return self.performanceRecordSum

    def getAveragePerformanceRecordAverage(self):
        if not self.didPerformAveraging:
            raise Exception('Must average before averaging the average')
        acc = 0
        count = 0
        for taskKey in self.performanceRecordSum:
            for idx in range(len(self.performanceRecordSum[taskKey])):
                if self.performanceRecordSum[taskKey][idx] != None:
                    acc += self.performanceRecordSum[taskKey][idx]
                    count += 1
        return acc / count
