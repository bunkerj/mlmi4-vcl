import sys
sys.path.append('../')

from visualization import Visualization

# fileNames = ["exp_3/SN_VCL.p", "exp_3/SN_VCL_RC_40.p", "exp_3/SN_VCL_RCO_40.p", "exp_3/SN_VCL_KC_40.p", "exp_3/SN_VCL_KCO_40.p"]

fileNames = ["../averaged_results/exp_3/SN_VCL.p", "../averaged_results/exp_3/SN_VCL_RC_40.p", "../averaged_results/exp_3/SN_VCL_RC_80.p",\
 "../averaged_results/exp_3/SN_VCL_RC_160.p",  "../averaged_results/exp_3/SN_VCL_RC_320.p", "../averaged_results/exp_3/SN_VCL_KC_40.p",\
 "../averaged_results/exp_3/SN_VCL_KC_40.p", "../averaged_results/exp_3/SN_VCL_KC_80.p", "../averaged_results/exp_3/SN_VCL_KC_160.p",\
 "../averaged_results/exp_3/SN_VCL_KC_320.p"]
expNames = ['VCL (No Coreset)', 'VCL + Random Coreset', 'VCL + Random Coreset Only', 'VCL + K-Center Coreset', 'VCL + K-Center Coreset Only']
colors = ['k','r','r','b','b']
styles = ['o-','o-','o--','o-','o--']

taskNames = {}
taskNames[0] = 'Task 0 (A or B)'
taskNames[1] = 'Task 1 (C or D)'
taskNames[2] = 'Task 2 (E or F)'
taskNames[3] = 'Task 3 (G or H)'
taskNames[4] = 'Task 4 (I or J)'

taskOrder = list(range(5))

visualization = Visualization(fileNames, expNames, taskNames, taskOrder, colors, styles)
visualization.plotTaskPerformance(numRows = 2, displayAvg = True)
#visualization.plotAvgPerformance()
