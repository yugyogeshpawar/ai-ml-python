# introduction_to_numpy_example.py

import numpy as np
import time

# --- The NumPy Advantage: Performance ---

# Let's compare the time it takes to sum the squares of a large sequence of numbers
# using a standard Python list vs. a NumPy array.

# Define the size of the dataset
num_elements = 1_000_000

# --- Method 1: Standard Python List ---
print("--- Using Standard Python List ---")
start_time_list = time.time()

# Create a list of numbers
python_list = list(range(num_elements))

# Square each number and sum the results
sum_of_squares_list = sum([x**2 for x in python_list])

end_time_list = time.time()
duration_list = end_time_list - start_time_list

print(f"Number of elements: {num_elements}")
print(f"Sum of squares: {sum_of_squares_list}")
print(f"Time taken with Python list: {duration_list:.6f} seconds")
print("-" * 30)


# --- Method 2: NumPy Array ---
print("\n--- Using NumPy Array ---")
start_time_numpy = time.time()

# Create a NumPy array
numpy_array = np.arange(num_elements, dtype=np.int64) # Use 64-bit integer for large sums

# Square the array and sum the results using vectorized operations
# np.square is a ufunc that operates element-wise
# .sum() is a highly optimized aggregation method
sum_of_squares_numpy = np.sum(np.square(numpy_array))

end_time_numpy = time.time()
duration_numpy = end_time_numpy - start_time_numpy

print(f"Number of elements: {num_elements}")
print(f"Sum of squares: {sum_of_squares_numpy}")
print(f"Time taken with NumPy array: {duration_numpy:.6f} seconds")
print("-" * 30)


# --- Conclusion ---
print("\n--- Performance Comparison ---")
if duration_numpy < duration_list:
    performance_gain = duration_list / duration_numpy
    print(f"NumPy was {performance_gain:.2f} times faster than the standard Python list.")
else:
    print("NumPy was not faster in this case. This can happen for very small arrays.")

print("\nThis example clearly demonstrates the power of vectorization.")
print("NumPy performs the calculation on the entire array at once, avoiding slow Python loops.")
