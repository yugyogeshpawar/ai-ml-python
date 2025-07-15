# creating_numpy_arrays_example.py

import numpy as np

# --- 1. Creating Arrays from Python Lists ---
print("--- From Python Lists ---")

# Create a 1D array
list_1d = [10, 20, 30, 40]
array_1d = np.array(list_1d)
print("1D Array from list:", array_1d)
print("Type of array_1d:", type(array_1d))

# Create a 2D array (matrix)
list_2d = [[1, 2, 3], [4, 5, 6]]
array_2d = np.array(list_2d)
print("\n2D Array from list of lists:\n", array_2d)

# It's important that inner lists have the same length for a regular shape
list_irregular = [[1, 2], [3, 4, 5]]
# This will create an array of objects (lists), not a 2D array
array_irregular = np.array(list_irregular, dtype=object)
print("\nIrregular list creates an array of objects:", array_irregular)
print("-" * 30)


# --- 2. Creating Placeholder Arrays ---
print("\n--- Placeholder Arrays ---")

# Create a 1D array of 4 zeros
zeros_array = np.zeros(4)
print("1D Array of zeros:", zeros_array)

# Create a 2x3 array of ones
ones_array = np.ones((2, 3))
print("\n2x3 Array of ones:\n", ones_array)

# You can also create an uninitialized array with np.empty.
# Its contents are random and depend on the state of memory.
# This is slightly faster if you plan to fill every element yourself.
empty_array = np.empty((2, 2))
print("\n2x2 Empty (uninitialized) array:\n", empty_array)
print("-" * 30)


# --- 3. Creating Arrays with Sequences ---
print("\n--- Sequence Arrays ---")

# Create an array with values from 0 to 14
arange_array = np.arange(15)
print("Array from arange(15):", arange_array)

# Create an array from 5 to 15 with a step of 2.5
arange_float = np.arange(5, 16, 2.5)
print("Array from arange(5, 16, 2.5):", arange_float)

# Create 9 evenly spaced numbers between 1 and 100
linspace_array = np.linspace(1, 100, 9)
print("\nArray from linspace(1, 100, 9):\n", linspace_array)

# linspace is great for generating coordinates for plots
x_coords = np.linspace(-5, 5, 11)
print("\nCoordinates for a plot:", x_coords)
print("-" * 30)

# --- Project from Lesson Plan ---
# A script that creates and inspects various NumPy arrays,
# reporting their shape, data type, and dimensions.
print("\n--- Lesson Project ---")
# We will cover .shape, .dtype, and .ndim in the next lesson,
# but here is a preview.
project_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Created Project Array:\n", project_array)
# print("Shape:", project_array.shape)
# print("Data Type:", project_array.dtype)
# print("Number of Dimensions:", project_array.ndim)
