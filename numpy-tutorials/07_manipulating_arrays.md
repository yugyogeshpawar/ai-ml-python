# 7. Manipulating Arrays

In the previous tutorial, we explored advanced indexing techniques like Boolean and Fancy Indexing. Now, we'll learn how to manipulate the structure of NumPy arrays, which is crucial for preparing data for various AI/ML models. This includes reshaping, stacking, and splitting arrays.

## 7.1 Reshaping Arrays

Reshaping allows you to change the shape (dimensions) of an array without changing its data. The new shape must be compatible with the original number of elements.

### `np.reshape()`: Changing the shape of an array

```python
import numpy as np

arr = np.arange(1, 13) # Create a 1D array from 1 to 12
print("Original 1D Array:", arr)
print("Original Shape:", arr.shape) # Output: (12,)

# Reshape to a 2D array (3 rows, 4 columns)
reshaped_2d = arr.reshape(3, 4)
print("\nReshaped to 3x4 Matrix:\n", reshaped_2d)
print("New Shape:", reshaped_2d.shape) # Output: (3, 4)

# Reshape to a 2D array (4 rows, 3 columns)
reshaped_2d_alt = arr.reshape(4, 3)
print("\nReshaped to 4x3 Matrix:\n", reshaped_2d_alt)

# Reshape to a 3D array (2 "layers", 3 rows, 2 columns)
reshaped_3d = arr.reshape(2, 3, 2)
print("\nReshaped to 2x3x2 Tensor:\n", reshaped_3d)
print("New Shape:", reshaped_3d.shape) # Output: (2, 3, 2)
```

**Using `-1` for unknown dimension:**
You can specify `-1` for one of the dimensions, and NumPy will automatically calculate the correct size for that dimension based on the total number of elements.

```python
import numpy as np

arr = np.arange(1, 13)

# Reshape to 3 rows, let NumPy figure out columns
reshaped_auto_cols = arr.reshape(3, -1)
print("\nReshaped to 3 rows (auto columns):\n", reshaped_auto_cols)
print("New Shape:", reshaped_auto_cols.shape) # Output: (3, 4)

# Reshape to 2 columns, let NumPy figure out rows
reshaped_auto_rows = arr.reshape(-1, 2)
print("\nReshaped to 2 columns (auto rows):\n", reshaped_auto_rows)
print("New Shape:", reshaped_auto_rows.shape) # Output: (6, 2)
```

### `np.newaxis` or `np.expand_dims()`: Adding a new dimension

These are useful for converting a 1D array into a row or column vector, which is often necessary for broadcasting or matrix operations.

```python
import numpy as np

arr_1d = np.array([1, 2, 3, 4])
print("Original 1D Array:", arr_1d)
print("Original Shape:", arr_1d.shape) # Output: (4,)

# Convert to a row vector (1 row, 4 columns)
row_vector = arr_1d[np.newaxis, :]
print("\nRow Vector (using newaxis):\n", row_vector)
print("Shape:", row_vector.shape) # Output: (1, 4)

# Convert to a column vector (4 rows, 1 column)
col_vector = arr_1d[:, np.newaxis]
print("\nColumn Vector (using newaxis):\n", col_vector)
print("Shape:", col_vector.shape) # Output: (4, 1)

# Using np.expand_dims()
expanded_row = np.expand_dims(arr_1d, axis=0) # Add dimension at axis 0 (for rows)
print("\nExpanded Row (using expand_dims):\n", expanded_row)
print("Shape:", expanded_row.shape) # Output: (1, 4)

expanded_col = np.expand_dims(arr_1d, axis=1) # Add dimension at axis 1 (for columns)
print("Expanded Column (using expand_dims):\n", expanded_col)
print("Shape:", expanded_col.shape) # Output: (4, 1)
```

### `np.squeeze()`: Removing single-dimensional entries

This is the opposite of `expand_dims`. It removes dimensions of size 1.

```python
import numpy as np

arr_squeezable = np.array([[[1, 2, 3]]]) # Shape (1, 1, 3)
print("Original Squeezable Array:\n", arr_squeezable)
print("Original Shape:", arr_squeezable.shape) # Output: (1, 1, 3)

squeezed_arr = np.squeeze(arr_squeezable)
print("\nSqueezed Array:", squeezed_arr)
print("New Shape:", squeezed_arr.shape) # Output: (3,)
```

## 7.2 Stacking Arrays

Stacking combines multiple arrays along a new or existing axis.

### `np.vstack()`: Stack arrays vertically (row-wise)

Arrays must have the same number of columns.

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stacking 1D arrays vertically creates a 2D array
stacked_1d_vert = np.vstack((arr1, arr2))
print("\nStacked 1D Arrays Vertically:\n", stacked_1d_vert)
print("Shape:", stacked_1d_vert.shape) # Output: (2, 3)

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Stacking 2D arrays vertically
stacked_2d_vert = np.vstack((matrix1, matrix2))
print("\nStacked 2D Arrays Vertically:\n", stacked_2d_vert)
print("Shape:", stacked_2d_vert.shape) # Output: (4, 2)
```

### `np.hstack()`: Stack arrays horizontally (column-wise)

Arrays must have the same number of rows.

```python
import numpy as np

arr1 = np.array([1, 2, 3])[:, np.newaxis] # Make them column vectors
arr2 = np.array([4, 5, 6])[:, np.newaxis]

# Stacking 1D column vectors horizontally creates a 2D array
stacked_1d_horiz = np.hstack((arr1, arr2))
print("\nStacked 1D Column Vectors Horizontally:\n", stacked_1d_horiz)
print("Shape:", stacked_1d_horiz.shape) # Output: (3, 2)

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Stacking 2D arrays horizontally
stacked_2d_horiz = np.hstack((matrix1, matrix2))
print("\nStacked 2D Arrays Horizontally:\n", stacked_2d_horiz)
print("Shape:", stacked_2d_horiz.shape) # Output: (2, 4)
```

### `np.concatenate()`: General stacking function

`np.concatenate()` allows more general stacking along a specified axis.

```python
import numpy as np

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0 (rows) - equivalent to vstack
concat_axis0 = np.concatenate((arr1, arr2), axis=0)
print("\nConcatenated along axis 0:\n", concat_axis0)

# Concatenate along axis 1 (columns) - equivalent to hstack
concat_axis1 = np.concatenate((arr1, arr2), axis=1)
print("\nConcatenated along axis 1:\n", concat_axis1)

# Concatenate 3D arrays
tensor1 = np.arange(1, 5).reshape(1, 2, 2) # Shape (1, 2, 2)
tensor2 = np.arange(5, 9).reshape(1, 2, 2) # Shape (1, 2, 2)

print("\nTensor 1:\n", tensor1)
print("Tensor 2:\n", tensor2)

# Concatenate along axis 0 (depth)
concat_3d_axis0 = np.concatenate((tensor1, tensor2), axis=0)
print("\nConcatenated 3D along axis 0:\n", concat_3d_axis0)
print("Shape:", concat_3d_axis0.shape) # Output: (2, 2, 2)
```

## 7.3 Splitting Arrays

Splitting divides a single array into multiple smaller arrays.

### `np.vsplit()`: Split arrays vertically (row-wise)

```python
import numpy as np

matrix = np.arange(1, 17).reshape(4, 4)
print("Original Matrix for splitting:\n", matrix)

# Split into 2 equal parts vertically
split_vertically = np.vsplit(matrix, 2)
print("\nSplit Vertically into 2 parts:")
for part in split_vertically:
    print(part)
    print("---")
# Output:
# [[ 1  2  3  4]
#  [ 5  6  7  8]]
# ---
# [[ 9 10 11 12]
#  [13 14 15 16]]
# ---

# Split at specific row indices
split_at_indices = np.vsplit(matrix, [1, 3]) # Split before row 1, and before row 3
print("\nSplit Vertically at indices [1, 3]:")
for part in split_at_indices:
    print(part)
    print("---")
# Output:
# [[1 2 3 4]]
# ---
# [[ 5  6  7  8]
#  [ 9 10 11 12]]
# ---
# [[13 14 15 16]]
# ---
```

### `np.hsplit()`: Split arrays horizontally (column-wise)

```python
import numpy as np

matrix = np.arange(1, 17).reshape(4, 4)
print("Original Matrix for splitting:\n", matrix)

# Split into 4 equal parts horizontally
split_horizontally = np.hsplit(matrix, 4)
print("\nSplit Horizontally into 4 parts:")
for part in split_horizontally:
    print(part)
    print("---")
# Output:
# [[ 1]
#  [ 5]
#  [ 9]
#  [13]]
# ---
# [[ 2]
#  [ 6]
#  [10]
#  [14]]
# ---
# [[ 3]
#  [ 7]
#  [11]
#  [15]]
# ---
# [[ 4]
#  [ 8]
#  [12]
#  [16]]
# ---
```

### `np.array_split()`: General splitting function

`np.array_split()` allows splitting into an arbitrary number of sections, even if they are not of equal size.

```python
import numpy as np

arr = np.arange(1, 8) # 7 elements
print("Original Array for array_split:", arr)

# Split into 3 parts (will be unequal)
split_unequal = np.array_split(arr, 3)
print("\nSplit into 3 unequal parts:")
for part in split_unequal:
    print(part)
# Output:
# [1 2 3]
# [4 5]
# [6 7]
```

## Practical Applications in AI/ML

*   **Data Preprocessing:** Reshaping data to fit the input requirements of models (e.g., converting 1D image data to 2D or 3D).
*   **Batching:** Splitting large datasets into smaller batches for training neural networks.
*   **Feature Engineering:** Combining different feature sets (arrays) into a single dataset.
*   **Model Output Handling:** Reshaping model predictions to match desired output formats.

## Assignment: Array Manipulation Practice

1.  Create a 1D array `data` with 24 elements (e.g., using `np.arange`).
2.  Reshape `data` into a 2D array of shape (6, 4). Print the reshaped array and its shape.
3.  Reshape the 2D array from step 2 into a 3D array of shape (2, 3, 4). Print the 3D array and its shape.
4.  Create two 2D arrays, `matrix_A` (shape 2x3) and `matrix_B` (shape 2x3), filled with random integers.
5.  Vertically stack `matrix_A` and `matrix_B`. Print the result and its shape.
6.  Horizontally stack `matrix_A` and `matrix_B`. Print the result and its shape.
7.  Create a 2D array `large_matrix` of shape (6, 6) with random integers.
8.  Split `large_matrix` into 3 equal parts horizontally. Print each part.

---

In the next tutorial, we will explore Linear Algebra operations with NumPy, which are fundamental to many AI/ML algorithms.

**Next:** [8. Linear Algebra with NumPy](08_linear_algebra_with_numpy.md)
**Previous:** [6. Advanced Indexing](06_advanced_indexing.md)
