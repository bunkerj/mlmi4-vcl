from visualization import Visualization

fileNames = ["exp_1/PM_VCL.p"]
expNames = ['VCL (No Coreset)']
colors = ['k']
styles = ['o-']

sizes = [200, 400, 1000, 2500, 5000]
for i in range(len(sizes)):
    fileNames.append('exp_1/PM_VCL_RC_{}.p'.format(sizes[i]))
    expNames.append('VCL + Random Coreset ({})'.format(sizes[i]))
    colors.append((i/len(sizes), 0, 1-i/len(sizes)))
    styles.append('o-')
for i in range(len(sizes)):
    fileNames.append('exp_1/PM_VCL_RCO_{}.p'.format(sizes[i]))
    expNames.append('Random Coreset Only ({})'.format(sizes[i]))
    colors.append((i/len(sizes), 0, 1-i/len(sizes)))
    styles.append('o--')

taskNames = {}
for i in range(5):
    taskNames[i] = 'Task {}'.format(i)
taskOrder = list(range(5))

visualization = Visualization(fileNames, expNames, taskNames, taskOrder, colors, styles)
#visualization.plotTaskPerformance(numRows = 2, displayAvg = True)
visualization.plotAvgPerformance()
