import pickle
import math
import matplotlib.pylab as plt

class Visualization:
    def __init__(self, fileNames, expNames, taskNames, taskOrder, colors, styles):
        self.fileNames = fileNames
        self.expNames = expNames
        self.taskNames = taskNames  #dictionary
        self.taskOrder = taskOrder  #list
        self.styles = styles
        self.colors = colors
        self.accList = self.readPickles(self.fileNames)

    def readPickles(self, filenames):
        accList = []
        for filename in filenames:
            acc = pickle.load(open(filename, 'rb'))
            accList.append(acc)
        return accList

    # plots the accuracy of each task at each time of arrival
    def plotTaskPerformance(self, numRows = 1, displayAvg = True):
        numCols = math.ceil((len(self.taskNames.keys()) + displayAvg) / numRows)
        plt.figure(figsize=(5*numCols, 5*numRows))
        plt.rcParams.update({'font.size': 20})

        idx = 1
        for taskId in self.taskNames.keys():
            plt.subplot(numRows, numCols, idx)
            for i in range(len(self.expNames)):
                acc = self.accList[i][taskId]
                expName = self.expNames[i]
                plt.plot(acc, self.styles[i], color=self.colors[i], linewidth=3, label=expName)

            plt.xticks(range(len(self.taskOrder)), self.taskOrder)
            plt.xlabel('Task Arrival')
            plt.ylim(0,1)
            if idx == 1 or idx == 4:
                plt.ylabel('Accuracy')
            else:
                plt.tick_params(labelleft=False)
            plt.grid(True)
            plt.title(self.taskNames[taskId])
            idx += 1

        if displayAvg == True:
            plt.subplot(numRows, numCols, idx)
            self.plotAvgPerformance(average_only = False)
            plt.tick_params(labelleft=False)

        #plt.legend(loc='upper center', fancybox=True, shadow=True, ncol=5)
        plt.tight_layout()
        #plt.legend(loc='upper center', bbox_to_anchor=(-3, 1.5), ncol = len(self.expNames))
        plt.savefig('test_task.png')
        print('done')

    def getAvgAccuracy(self, accuracy):
        avgAccuracy = []
        for i in range(len(self.taskOrder)):
            acc = 0
            count = 0
            for taskId in self.taskNames.keys():
                if accuracy[taskId][i] is not None:
                    acc += accuracy[taskId][i]
                    count += 1
            avgAccuracy.append(acc / count)
        return avgAccuracy

    # plots average accuracy at each time of arrival
    def plotAvgPerformance(self, average_only = True):
        if average_only == True:
            plt.figure(figsize=(10,8))
            plt.ylabel('Accuracy')

        for i in range(len(self.expNames)):
            acc = self.accList[i]
            avgAccuracy = self.getAvgAccuracy(acc)
            expName = self.expNames[i]
            plt.plot(avgAccuracy, self.styles[i], color=self.colors[i], linewidth=3, label=expName)

        plt.xticks(range(len(self.taskOrder)), self.taskOrder)
        plt.xlabel('Task Arrival')
        plt.grid()
        plt.title('Average')
        #plt.legend(loc='upper center', ncol = len(self.expNames))
        plt.savefig('test_avg.png')
