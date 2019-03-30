import sys
sys.path.append('../')
sys.path.append('../exp/')

from result_averager import ResultAverager

def isPerformanceRecordEquals(pr1, pr2):
    for taskKey in pr1:
        for idx in range(len(pr1[taskKey])):
            if pr1[taskKey][idx] != pr2[taskKey][idx]:
                return False
    return True

performanceRecord1 = {
'0': [2, 4, 6, 8],
'1': [None, 4, 6, 8],
'2': [None, None, 6, 8],
'3': [None, None, None, 8],
}

performanceRecord2 = {
'0': [8, 6, 4, 2],
'1': [None, 6, 4, 2],
'2': [None, None, 4, 2],
'3': [None, None, None, 2],
}

expectedRecord = {
'0': [5, 5, 5, 5],
'1': [None, 5, 5, 5],
'2': [None, None, 5, 5],
'3': [None, None, None, 5],
}

averager = ResultAverager()

averager.add(performanceRecord1)
averager.add(performanceRecord2)

averagePerformanceRecord = averager.getAveragePerformanceRecord()

assert isPerformanceRecordEquals(averagePerformanceRecord, expectedRecord)
assert averager.getAveragePerformanceRecordAverage() == 5
