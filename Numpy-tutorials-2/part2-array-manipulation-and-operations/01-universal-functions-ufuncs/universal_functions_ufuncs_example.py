# universal_functions_ufuncs_example.py

import numpy as np

# --- 1. Unary ufuncs (operate on one array) ---
print("--- Unary Ufuncs ---")
x = np.array([1, 2.5, 3.6, -4.2, -5.8])
print("Original array:", x)

# Mathematical functions
print("Square root (np.sqrt):", np.sqrt(np.abs(x))) # Use abs to avoid sqrt of negative
print("Exponential (np.exp):", np.exp(x))
print("Square (np.square):", np.square(x))

# Rounding functions
print("\nRounding:")
print("Floor (np.floor):", np.floor(x))
print("Ceiling (np.ceil):", np.ceil(x))

# Sign function
print("\nSign (np.sign):", np.sign(x)) # Returns -1 for negative, 0 for zero, 1 for positive

# Handling non-numerical values
y = np.array([1, 2, np.nan, 4, np.inf])
print("\nArray with NaN and Inf:", y)
print("Is NaN? (np.isnan):", np.isnan(y))
print("Is finite? (np.isfinite):", np.isfinite(y))
print("-" * 30)


# --- 2. Binary ufuncs (operate on two arrays) ---
print("\n--- Binary Ufuncs ---")
a = np.array([1, 5, 10, 15])
b = np.array([1, 2, 5, 5])
print("Array 'a':", a)
print("Array 'b':", b)

# Basic arithmetic (can use operators or function names)
print("\nArithmetic:")
print("Add (a + b):", a + b)
print("Subtract (a - b):", a - b)
print("Multiply (a * b):", a * b)
print("Divide (a / b):", a / b)
print("Power (a ** b):", a ** b)

# Comparison functions
print("\nComparisons:")
print("Greater (a > b):", np.greater(a, b))
print("Equal (a == b):", a == b)

# Element-wise max/min
print("\nMax/Min:")
print("Maximum (np.maximum):", np.maximum(a, b)) # Chooses the larger value from a or b at each position
print("Minimum (np.minimum):", np.minimum(a, b))
print("-" * 30)


# --- 3. Ufuncs can have optional 'out' arguments ---
print("\n--- Ufunc 'out' Argument ---")
# This allows you to perform computations in-place, without creating a new array.
# This can save memory for very large arrays.

z = np.zeros(4)
print("Array 'z' before:", z)
np.add(a, b, out=z) # The result of a + b is stored directly in z
print("Array 'z' after np.add(a, b, out=z):", z)

# The original arrays 'a' and 'b' are unchanged
print("Array 'a' is unchanged:", a)
print("Array 'b' is unchanged:", b)
print("-" * 30)
