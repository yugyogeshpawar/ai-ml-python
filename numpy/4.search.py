import numpy as np

arr = np.array([1,4,5,6,7,33,3])

newarr = np.array_split(arr, 2)

index = np.where(newarr[1] == 33)

print(index)







# index = np.where(arr == 33)

# print(index)

# for i, x in enumerate(arr):
#     if x == 33:
#         print(i)


