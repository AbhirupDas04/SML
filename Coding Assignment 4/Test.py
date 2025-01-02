import numpy as np

arr1 = np.array([1,2,3])
arr2 = []
arr2.append(arr1.copy())
arr1[2] = 5
if np.array_equal(arr1, arr2[0]):
    print("Hey")
arr1[2] = 3
if np.array_equal(arr1, arr2[0]):
    print("Hey")