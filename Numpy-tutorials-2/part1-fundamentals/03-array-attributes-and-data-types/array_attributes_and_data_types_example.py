# array_attributes_and_data_types_example.py

import numpy as np

# --- 1. Inspecting Basic Array Attributes ---
print("--- Basic Array Attributes ---")

# Let's create a 3D array to see these attributes in action
# Shape: (2 layers, 3 rows, 4 columns)
array_3d = np.array([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
])

print("The Array:\n", array_3d)

# .ndim: Number of dimensions
print(f"\nNumber of dimensions (ndim): {array_3d.ndim}")

# .shape: The size of the array in each dimension
print(f"Shape of the array (shape): {array_3d.shape}")

# .size: The total number of elements in the array
print(f"Total number of elements (size): {array_3d.size}")

# Verify that size is the product of shape dimensions
rows, cols, depth = array_3d.shape
print(f"Is size equal to rows * cols * depth? {array_3d.size == rows * cols * depth}")
print("-" * 30)


# --- 2. Understanding Data Types (dtype) ---
print("\n--- Data Types (dtype) ---")

# NumPy automatically infers the data type upon creation
int_array = np.array([1, 2, 3])
print(f"\nArray: {int_array}")
print(f"Data type of int_array: {int_array.dtype}")

# If we include a float, the entire array is upcast to float
float_array = np.array([1.0, 2, 3])
print(f"\nArray: {float_array}")
print(f"Data type of float_array: {float_array.dtype}")
print("-" * 30)


# --- 3. Specifying and Controlling Data Types ---
print("\n--- Specifying Data Types ---")

# You can explicitly set the data type for memory efficiency or precision.
# Let's create an array of numbers that could fit in an 8-bit integer (0-255)
# Default would be int64 (8 bytes per element)
default_type_arr = np.array([10, 20, 255])
print(f"\nArray with default dtype: {default_type_arr}")
print(f"Default dtype: {default_type_arr.dtype}")
print(f"Memory size (bytes): {default_type_arr.nbytes}") # .nbytes = .size * .itemsize

# Now let's use unsigned 8-bit integer
# uint8 can hold values from 0 to 255
uint8_arr = np.array([10, 20, 255], dtype=np.uint8)
print(f"\nArray with uint8 dtype: {uint8_arr}")
print(f"uint8 dtype: {uint8_arr.dtype}")
print(f"Memory size (bytes): {uint8_arr.nbytes}")

# What happens if you try to store a number that's too big?
# NumPy will "wrap around" (overflow) without warning!
overflow_arr = np.array([255, 256, 257], dtype=np.uint8)
print(f"\nOverflow example with uint8: {overflow_arr}") # 256 becomes 0, 257 becomes 1
print("This demonstrates the importance of choosing the correct dtype!")
print("-" * 30)


# --- Project from Lesson Plan ---
# A script that creates and inspects various NumPy arrays,
# reporting their shape, data type, and dimensions.
print("\n--- Lesson Project ---")
# This script itself serves as the project.
# We created a 3D array and inspected its ndim, shape, and size.
# We also created arrays with different dtypes and checked their memory usage.
# This covers the core concepts of the lesson.
print("Project complete: We have created and inspected arrays.")
