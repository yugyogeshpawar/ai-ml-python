# reshaping_and_transposing_example.py

import numpy as np

# --- 1. Reshaping Arrays ---
print("--- Reshaping Arrays ---")
arr = np.arange(24)
print("Original 1D array (size {}):".format(arr.size), arr)

# Reshape to a 4x6 matrix
reshaped_4x6 = arr.reshape(4, 6)
print("\nReshaped to (4, 6):\n", reshaped_4x6)

# Reshape to a 2x3x4 3D array
reshaped_3d = arr.reshape(2, 3, 4)
print("\nReshaped to (2, 3, 4):\n", reshaped_3d)

# Using -1 to infer a dimension
# Reshape to have 8 rows, let NumPy figure out the columns
reshaped_neg_one = arr.reshape(8, -1)
print("\nReshaped to (8, -1), inferred shape is {}:\n{}".format(
    reshaped_neg_one.shape, reshaped_neg_one
))

# Trying an incompatible shape will raise an error
try:
    arr.reshape(5, 5)
except ValueError as e:
    print(f"\nAs expected, reshaping a size 24 array to (5, 5) fails:\n{e}")
print("-" * 30)


# --- 2. Flattening: ravel() vs flatten() ---
print("\n--- Flattening Arrays ---")
matrix = np.array([[10, 20, 30], [40, 50, 60]])
print("Original matrix:\n", matrix)

# Using ravel() (returns a view)
raveled_arr = matrix.ravel()
print("\nRaveled array:", raveled_arr)

# Modify the raveled array
raveled_arr[0] = 999
print("Original matrix after modifying raveled array:\n", matrix) # It changed!

# Using flatten() (returns a copy)
matrix = np.array([[10, 20, 30], [40, 50, 60]]) # Reset matrix
flattened_arr = matrix.flatten()
print("\nFlattened array:", flattened_arr)

# Modify the flattened array
flattened_arr[0] = 111
print("Original matrix after modifying flattened array:\n", matrix) # It did NOT change
print("-" * 30)


# --- 3. Transposing Arrays ---
print("\n--- Transposing Arrays ---")
matrix = np.arange(1, 10).reshape(3, 3)
print("Original matrix (3, 3):\n", matrix)

# Transpose using the .T attribute
transposed_matrix = matrix.T
print("\nTransposed matrix:\n", transposed_matrix)

# Transposing a non-square matrix
rect_matrix = np.arange(6).reshape(2, 3)
print("\nOriginal rectangular matrix (2, 3):\n", rect_matrix)
print("Transposed matrix (3, 2):\n", rect_matrix.T)

# Transposing is also a view
print("\nTransposing returns a view...")
transposed_view = rect_matrix.T
transposed_view[0, 0] = 100
print("Original matrix after modifying the transposed view:\n", rect_matrix)
print("-" * 30)
