import pickle
def readPickles(filenames):
    accList = []
    for filename in filenames:
        acc = pickle.load(open(filename, 'rb'))
        print(filename, end = "\n")
        print(acc,  end ="\n")
        accList.append(acc)
    return accList

print("MNIST")
fileNames = ["exp_2/SM_VCL.p", "exp_2/SM_VCL_RC_40.p", "exp_2/SM_VCL_KC_40.p",\
                "exp_2/SM_VCL_RC_80.p", "exp_2/SM_VCL_KC_80.p",  "exp_2/SM_VCL_RC_160.p", "exp_2/SM_VCL_KC_160.p", "exp_2/SM_VCL_RC_320.p", "exp_2/SM_VCL_KC_320.p"]
readPickles(fileNames)

# print("NOTMNIST")
# fileNames = ["exp_3/SN_VCL.p", "exp_3/SN_VCL_RC_40.p", "exp_3/SN_VCL_KC_40.p",\
#                 "exp_3/SN_VCL_RC_80.p", "exp_3/SN_VCL_KC_80.p",  "exp_3/SN_VCL_RC_160.p", "exp_3/SN_VCL_KC_160.p", "exp_3/SN_VCL_RC_320.p", "exp_3/SN_VCL_KC_320.p"]
# readPickles(fileNames)
