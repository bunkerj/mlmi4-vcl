from visualization import Visualization

fileNames = ["exp_2/SM_VCL.p", "exp_2/SM_VCL_RC_40.p", "exp_2/SM_VCL_RCO_40.p", "exp_2/SM_VCL_KC_40.p", "exp_2/SM_VCL_KCO_40.p"]
expNames = ['VCL (No Coreset)', 'VCL + Random Coreset', 'VCL + Random Coreset Only', 'VCL + K-Center Coreset', 'VCL + K-Center Coreset Only']
colors = ['k','r','r','b','b']
styles = ['o-','o-','o--','o-','o--']

taskNames = {}
taskNames[0] = 'Task 0 (0 or 1)'
taskNames[1] = 'Task 1 (2 or 3)'
taskNames[2] = 'Task 2 (4 or 5)'
taskNames[3] = 'Task 3 (6 or 7)'
taskNames[4] = 'Task 4 (8 or 9)'

taskOrder = list(range(5))

visualization = Visualization(fileNames, expNames, taskNames, taskOrder, colors, styles)
visualization.plotTaskPerformance(numRows = 2, displayAvg = True)
#visualization.plotAvgPerformance()
