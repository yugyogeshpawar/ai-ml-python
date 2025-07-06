# 3. Array Attributes and Basics

In the previous tutorial, we learned how to create NumPy arrays. Now, let's explore the fundamental attributes of these arrays and how to perform basic operations like indexing and slicing. Understanding these concepts is crucial for effectively manipulating data in AI/ML tasks.

## 3.1 Understanding Array Attributes

NumPy arrays come with several useful attributes that provide information about their structure and data type.

### `ndim`: Number of Dimensions

This attribute tells you the number of dimensions (axes) of the array.

```python
import numpy as np

# 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array ndim:", arr_1d.ndim) # Output: 1

# 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array ndim:", arr_2d.ndim) # Output: 2

# 3D array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3D Array ndim:", arr_3d.ndim) # Output: 3
```

### `shape`: Dimensions of the Array

The `shape` attribute returns a tuple indicating the size of the array in each dimension.

```python
import numpy as np

arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array shape:", arr_1d.shape) # Output: (5,)

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array shape:", arr_2d.shape) # Output: (2, 3) (2 rows, 3 columns)

arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3D Array shape:", arr_3d.shape) # Output: (2, 2, 2) (2 "layers", 2 rows, 2 columns)
```

### `size`: Total Number of Elements

This attribute returns the total number of elements in the array. It's equal to the product of the elements of the `shape` tuple.

```python
import numpy as np

arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array size:", arr_1d.size) # Output: 5

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array size:", arr_2d.size) # Output: 6 (2 * 3)
```

### `dtype`: Data Type of Elements

The `dtype` attribute describes the type of the elements in the array. NumPy automatically infers the data type when you create an array, but you can also specify it.

```python
import numpy as np

arr_int = np.array([1, 2, 3])
print("Integer Array dtype:", arr_int.dtype) # Output: int64 (or int32 depending on system)

arr_float = np.array([1.0, 2.5, 3.0])
print("Float Array dtype:", arr_float.dtype) # Output: float64

arr_mixed = np.array([1, 2.5, 3]) # NumPy will upcast to the most general type
print("Mixed Array dtype:", arr_mixed.dtype) # Output: float64

# Specifying dtype explicitly
arr_specified_dtype = np.array([1, 2, 3], dtype=np.float32)
print("Specified Dtype Array:", arr_specified_dtype)
print("Specified Dtype Array dtype:", arr_specified_dtype.dtype) # Output: float32
```

Common `dtype` values include `int64`, `float64`, `bool`, `complex128`, etc.

## 3.2 Indexing and Slicing

Accessing specific elements or subsets of an array is fundamental. NumPy's indexing and slicing work similarly to Python lists, but with extensions for multi-dimensional arrays.

### 3.2.1 1D Array Indexing and Slicing

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

# Accessing a single element (0-indexed)
print("\n1D Array Indexing:")
print("First element:", arr[0])    # Output: 10
print("Third element:", arr[2])    # Output: 30
print("Last element:", arr[-1])   # Output: 60

# Slicing (start:stop:step)
print("\n1D Array Slicing:")
print("Elements from index 1 to 3 (exclusive):", arr[1:4]) # Output: [20 30 40]
print("Elements from beginning to index 2 (exclusive):", arr[:2]) # Output: [10 20]
print("Elements from index 3 to end:", arr[3:]) # Output: [40 50 60]
print("All elements:", arr[:]) # Output: [10 20 30 40 50 60]
print("Every second element:", arr[::2]) # Output: [10 30 50]
print("Reversed array:", arr[::-1]) # Output: [60 50 40 30 20 10]
```

### 3.2.2 2D Array Indexing and Slicing

For 2D arrays (matrices), you use `[row_index, column_index]`.

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("\n2D Array Indexing and Slicing:")
print("Original Matrix:\n", matrix)

# Accessing a single element (row 0, column 1)
print("Element at (0, 1):", matrix[0, 1]) # Output: 2

# Accessing an entire row
print("First row:", matrix[0, :]) # Output: [1 2 3]
print("Second row:", matrix[1])   # Output: [4 5 6] (shorthand for matrix[1, :])

# Accessing an entire column
print("First column:", matrix[:, 0]) # Output: [1 4 7]
print("Third column:", matrix[:, 2]) # Output: [3 6 9]

# Slicing rows and columns
print("First two rows, first two columns:\n", matrix[:2, :2])
# Output:
# [[1 2]
#  [4 5]]

print("Last row, all columns:\n", matrix[-1, :]) # Output: [7 8 9]

print("All rows, last column:\n", matrix[:, -1]) # Output: [3 6 9]
```

### 3.2.3 3D Array Indexing and Slicing

For 3D arrays, the general format is `[depth_index, row_index, column_index]`. Think of `depth_index` as selecting a 2D slice (a "plane" or "layer") from the 3D array.

```python
import numpy as np

tensor = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])

print("\n3D Array Indexing and Slicing:")
print("Original Tensor:\n", tensor)
print("Shape:", tensor.shape) # Output: (2, 2, 3)

# Accessing a single element (first depth, first row, second column)
print("Element at (0, 0, 1):", tensor[0, 0, 1]) # Output: 2

# Accessing a 2D slice (the first "layer")
print("First 2D slice:\n", tensor[0, :, :])
# Output:
# [[1 2 3]
#  [4 5 6]]

# Accessing a row from a specific 2D slice
print("Second row of the second 2D slice:", tensor[1, 1, :]) # Output: [10 11 12]

# Accessing a column across all 2D slices
print("Second column across all layers and rows:\n", tensor[:, :, 1])
# Output:
# [[ 2  5]
#  [ 8 11]]
```

## 3.3 Basic Array Operations (Element-wise)

NumPy allows you to perform arithmetic operations directly on arrays. These operations are applied element-wise, meaning the operation is performed on corresponding elements of the arrays.

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

print("\nBasic Element-wise Operations:")
print("arr1:", arr1)
print("arr2:", arr2)

# Addition
print("Addition (arr1 + arr2):", arr1 + arr2) # Output: [ 6  8 10 12]

# Subtraction
print("Subtraction (arr1 - arr2):", arr1 - arr2) # Output: [-4 -4 -4 -4]

# Multiplication
print("Multiplication (arr1 * arr2):", arr1 * arr2) # Output: [ 5 12 21 32]

# Division
print("Division (arr1 / arr2):", arr1 / arr2) # Output: [0.2        0.33333333 0.42857143 0.5       ]

# Exponentiation
print("Exponentiation (arr1 ** 2):", arr1 ** 2) # Output: [ 1  4  9 16]

# Operations with a scalar
print("Scalar Addition (arr1 + 10):", arr1 + 10) # Output: [11 12 13 14]
print("Scalar Multiplication (arr1 * 2):", arr1 * 2) # Output: [ 2  4  6  8]
```

**Important Note:** For element-wise operations, the arrays must have compatible shapes. If they don't, NumPy's broadcasting rules (which we'll cover in a later tutorial) come into play.

## Assignment: Array Attributes and Operations

1.  Create a 2D array (matrix) of shape (4, 5) filled with random integers between 1 and 100.
2.  Print its `ndim`, `shape`, and `size`.
3.  Access and print the element in the 3rd row, 2nd column (remember 0-indexing!).
4.  Extract and print the first two rows of the matrix.
5.  Extract and print the last three columns of the matrix.
6.  Create another 2D array of the same shape (4, 5) filled with ones.
7.  Perform element-wise multiplication of your random array with the array of ones. What do you observe?

---

In the next tutorial, we will delve into Universal Functions (ufuncs) and their role in efficient array computations.

**Next:** [4. Universal Functions (ufuncs)](04_universal_functions_ufuncs.md)
**Previous:** [2. Creating NumPy Arrays](02_creating_numpy_arrays.md)
