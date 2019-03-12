from visualization import Visualization

fileNames = ["exp_1/PM_VCL.p", "exp_1/PM_VCL_RC_200.p", "exp_1/PM_VCL_RCO_200.p", "exp_1/PM_VCL_KC_200.p", "exp_1/PM_VCL_KCO_200.p"]
expNames = ['VCL (No Coreset)', 'VCL + Random Coreset', 'VCL + Random Coreset Only', 'VCL + K-Center Coreset', 'VCL + K-Center Coreset Only']
colors = ['k','r','r','b','b']
styles = ['o-','o-','o--','o-','o--']

taskNames = {}
for i in range(5):
    taskNames[i] = 'Task {}'.format(i)
taskOrder = list(range(5))

visualization = Visualization(fileNames, expNames, taskNames, taskOrder, colors, styles)
#visualization.plotTaskPerformance(numRows = 2, displayAvg = True)
visualization.plotAvgPerformance()
