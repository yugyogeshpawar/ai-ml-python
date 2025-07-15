# advanced_indexing_example.py

import numpy as np

# --- 1. Boolean Array Indexing ---
print("--- Boolean Array Indexing ---")
# Create an array of names and a parallel array of random data
names = np.array(['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Bob'])
data = np.random.randn(6, 4) # 6 rows, 4 columns

print("Names array:", names)
print("Data array:\n", data)

# Find all rows where the name is 'Alice'
# First, create the boolean condition
is_alice = (names == 'Alice')
print("\nBoolean condition (names == 'Alice'):", is_alice)

# Now, use this boolean array to select rows from the data array
print("\nData rows where name is 'Alice':\n", data[is_alice])

# You can also do this in one line
print("\nData rows where name is not 'Bob':\n", data[names != 'Bob'])

# Combining conditions: select rows for 'Bob' OR 'David'
bob_or_david = (names == 'Bob') | (names == 'David')
print("\nData for Bob or David:\n", data[bob_or_david])

# Modifying data with boolean indexing
# Set all negative values in the data array to 0
print("\nOriginal data:\n", data)
data[data < 0] = 0
print("Data after setting negative values to 0:\n", data)
print("-" * 30)


# --- 2. Integer Array Indexing (Fancy Indexing) ---
print("\n--- Fancy Indexing ---")
# Create a 2D array
arr_2d = np.arange(32).reshape((8, 4))
print("Original 2D array:\n", arr_2d)

# Select specific rows in a custom order
# Get rows 4, 2, 0, and 6
selected_rows = arr_2d[[4, 2, 0, 6]]
print("\nSelected rows [4, 2, 0, 6]:\n", selected_rows)

# Using negative indices
# Get the last three rows in reverse order
selected_rows_neg = arr_2d[[-1, -2, -3]]
print("\nSelected rows [-1, -2, -3]:\n", selected_rows_neg)

# Selecting elements from specific row/column coordinates
# Get elements at (1, 0), (5, 3), (7, 1)
rows = np.array([1, 5, 7])
cols = np.array([0, 3, 1])
selected_elements = arr_2d[rows, cols]
print(f"\nElements at coordinates ({rows}, {cols}): {selected_elements}")

# Modifying elements with fancy indexing
print("\nOriginal array before modification:\n", arr_2d)
# Set the values of the selected elements to 0
arr_2d[rows, cols] = 0
print("Array after setting selected elements to 0:\n", arr_2d)
print("-" * 30)


# --- 3. Combining Indexing Types ---
print("\n--- Combining Indexing Types ---")
# You can mix fancy indexing with slicing
arr_2d = np.arange(32).reshape((8, 4)) # Reset the array
print("Original 2D array:\n", arr_2d)

# Select a subset of rows, and for those rows, select a slice of columns
# Get rows 1, 5, 7 and from them, columns 0 and 1
subset = arr_2d[[1, 5, 7], :2]
print("\nRows [1, 5, 7], columns :2 (first two):\n", subset)

# Select a slice of rows, and for those rows, select a fancy index of columns
# Get rows 0-3, and from them, columns 3, 0, 1
subset2 = arr_2d[:4, [3, 0, 1]]
print("\nRows :4 (first four), columns [3, 0, 1]:\n", subset2)
print("-" * 30)
