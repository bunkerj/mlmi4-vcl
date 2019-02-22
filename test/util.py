import numpy as np

def convertTensorToNumpyArray(tensor):
    return tensor.cpu().detach().numpy()

def areTensorsEqual(tensor1, tensor2):
    arr1 = convertTensorToNumpyArray(tensor1)
    arr2 = convertTensorToNumpyArray(tensor2)
    return np.array_equal(arr1, arr2)

def areListOfTensorsEqual(list1, list2):
    if len(list1) != len(list2):
        return False
    for index in range(len(list1)):
        if not areTensorsEqual(list1[index], list2[index]):
            return False
    return True
