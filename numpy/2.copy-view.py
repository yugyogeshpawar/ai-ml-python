import numpy as np

arr = np.array([1,4,5,6,7,33,3])

array2 = arr.copy()
# array2 = arr.view()

array2[0] = 10

print(arr)
print(array2)