# basic_indexing_and_slicing_example.py

import numpy as np

# --- 1. Indexing and Slicing 1D Arrays ---
print("--- 1D Array Indexing and Slicing ---")
arr_1d = np.arange(0, 10) * 5
print("Original 1D Array:", arr_1d)

# Accessing single elements
print(f"Element at index 3: {arr_1d[3]}")
print(f"Second to last element: {arr_1d[-2]}")

# Basic slicing
print(f"Slice from index 2 to 5: {arr_1d[2:6]}")

# Slicing with assignment
print("\nModifying a slice of the 1D array...")
slice_to_modify = arr_1d[2:5]
print(f"Original slice: {slice_to_modify}")
slice_to_modify[:] = -1 # Use [:] to modify the contents of the slice in place
print(f"Original array after modification: {arr_1d}")

# Demonstrating that a slice is a view
print("\nDemonstrating slices are views...")
view_slice = arr_1d[6:8]
print(f"Slice (view): {view_slice}")
view_slice[0] = 999
print(f"Original array after modifying the view: {arr_1d}")

# Using .copy() to create a separate copy
print("\nUsing .copy() to avoid modifying the original...")
copy_slice = arr_1d[0:3].copy()
copy_slice[0] = 0
print(f"The copied slice: {copy_slice}")
print(f"The original array remains unchanged: {arr_1d}")
print("-" * 30)


# --- 2. Indexing and Slicing 2D Arrays ---
print("\n--- 2D Array Indexing and Slicing ---")
# Create a 4x5 matrix
arr_2d = np.arange(1, 21).reshape(4, 5)
print("Original 2D Array:\n", arr_2d)

# Accessing a single element (row 2, column 3)
element = arr_2d[2, 3]
print(f"\nElement at (2, 3): {element}")

# Accessing a specific row (row 1)
row = arr_2d[1] # or arr_2d[1, :]
print(f"Row at index 1: {row}")

# Accessing a specific column (column 4)
col = arr_2d[:, 4]
print(f"Column at index 4: {col}")

# Slicing a sub-matrix
# Get the 2x2 matrix from the bottom-right corner
sub_matrix = arr_2d[2:, 3:]
print("\nSub-matrix (bottom right 2x2):\n", sub_matrix)

# Modifying a 2D slice
print("\nModifying a 2D slice...")
arr_2d[0:2, 0:2] = 0
print("Array after setting top-left 2x2 to 0:\n", arr_2d)
print("-" * 30)


# --- 3. Indexing in 3D Arrays ---
print("\n--- 3D Array Indexing ---")
# Create a 2x3x4 array
arr_3d = np.arange(24).reshape(2, 3, 4)
print("Original 3D Array:\n", arr_3d)

# Get a single value (layer 1, row 0, column 2)
val_3d = arr_3d[1, 0, 2]
print(f"\nValue at (1, 0, 2): {val_3d}")

# Get a 2D slice (layer 0)
layer_0 = arr_3d[0] # or arr_3d[0, :, :]
print("\nLayer 0:\n", layer_0)

# Get a more complex slice
# From layer 1, get all rows, but only column 1
slice_3d = arr_3d[1, :, 1]
print(f"\nFrom layer 1, all rows, column 1: {slice_3d}")
