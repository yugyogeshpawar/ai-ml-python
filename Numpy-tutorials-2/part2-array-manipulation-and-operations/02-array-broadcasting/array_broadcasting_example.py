# array_broadcasting_example.py

import numpy as np

# --- 1. Broadcasting a Scalar ---
print("--- Broadcasting a Scalar ---")
# The simplest case of broadcasting
arr = np.array([10, 20, 30, 40])
scalar = 5

# The scalar is "stretched" to shape (4,) to match the array
result = arr + scalar
print(f"{arr} + {scalar} -> {result}")
print("-" * 30)


# --- 2. Broadcasting a 1D Array to a 2D Array ---
print("\n--- Broadcasting a 1D Array to a 2D Array ---")
matrix = np.ones((3, 4), dtype=int)
vector = np.arange(4)

print("Matrix (shape {}):\n{}".format(matrix.shape, matrix))
print("Vector (shape {}): {}".format(vector.shape, vector))

# Broadcasting rules in action:
# matrix.shape -> (3, 4)
# vector.shape -> ( , 4)  (Rule 1: Padded) -> (1, 4)
# Dimension 1: 3 vs 1 -> 1 is stretched to 3
# Dimension 2: 4 vs 4 -> Match
# Resulting shape is (3, 4)

result_matrix = matrix + vector
print("\nResult of matrix + vector (shape {}):\n{}".format(result_matrix.shape, result_matrix))
print("-" * 30)


# --- 3. Broadcasting a Column Vector ---
print("\n--- Broadcasting a Column Vector ---")
matrix = np.arange(1, 10).reshape(3, 3)
# To broadcast down the columns, the smaller array must have shape (n, 1)
col_vector = np.array([[10], [20], [30]]) # Shape (3, 1)

print("Matrix (shape {}):\n{}".format(matrix.shape, matrix))
print("Column Vector (shape {}):\n{}".format(col_vector.shape, col_vector))

# Broadcasting rules in action:
# matrix.shape     -> (3, 3)
# col_vector.shape -> (3, 1)
# Dimension 1: 3 vs 3 -> Match
# Dimension 2: 3 vs 1 -> 1 is stretched to 3
# Resulting shape is (3, 3)

result_matrix = matrix + col_vector
print("\nResult of matrix + column vector (shape {}):\n{}".format(result_matrix.shape, result_matrix))
print("-" * 30)


# --- 4. A Case of Incompatible Shapes ---
print("\n--- Incompatible Shapes ---")
matrix = np.zeros((3, 4))
row_vector = np.arange(3) # Shape (3,)

print("Matrix shape:", matrix.shape)
print("Row vector shape:", row_vector.shape)

# Broadcasting rules analysis:
# matrix.shape     -> (3, 4)
# row_vector.shape -> ( , 3)  (Rule 1: Padded) -> (1, 3)
# Dimension 1: 3 vs 1 -> 1 is stretched to 3
# Dimension 2: 4 vs 3 -> MISMATCH! And neither is 1.
# This will raise a ValueError.

try:
    matrix + row_vector
except ValueError as e:
    print(f"\nAs expected, a ValueError was raised:\n{e}")

# To fix this, the vector needs to be a column vector with shape (3, 1)
# We can do this with np.newaxis
fix_vector = row_vector[:, np.newaxis] # Shape becomes (3, 1)
print("\nShape of fixed vector:", fix_vector.shape)
fixed_result = matrix + fix_vector
print("Result after fixing vector shape:\n", fixed_result)
print("-" * 30)
