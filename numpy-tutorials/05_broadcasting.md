# 5. Broadcasting

In the previous tutorial, we explored Universal Functions (ufuncs) and aggregation methods. Now, we'll delve into a powerful and often misunderstood feature of NumPy: **Broadcasting**. Broadcasting allows NumPy to perform operations on arrays of different shapes, making your code more concise and efficient.

## What is Broadcasting?

Broadcasting is the ability of NumPy to treat arrays of different shapes during arithmetic operations. When performing an operation on two arrays, NumPy compares their shapes element-wise. It then "stretches" or "broadcasts" the smaller array across the larger array so that they have compatible shapes. This happens without actually creating copies of the data, which makes it very memory-efficient.

## Broadcasting Rules

For two arrays to be compatible for broadcasting, their dimensions must satisfy one of two conditions, starting from the trailing (rightmost) dimension:

1.  **Equal:** The dimensions are equal.
2.  **One:** One of the dimensions is 1.

If neither of these conditions is met, a `ValueError` will be raised.

Let's illustrate with examples.

### Example 1: Scalar and Array

This is the simplest form of broadcasting. A scalar (a single number) can be broadcast across an entire array.

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
scalar = 5

# Adding a scalar to an array
result_add = arr + scalar
print("Array + Scalar:", result_add) # Output: [ 6  7  8  9]

# Multiplying an array by a scalar
result_mul = arr * scalar
print("Array * Scalar:", result_mul) # Output: [ 5 10 15 20]
```

**Explanation:** The scalar `5` is effectively "stretched" to match the shape of `arr` `(4,)`, so it behaves as if you were adding `[5, 5, 5, 5]` to `arr`.

### Example 2: 1D Array and 2D Array

Consider adding a 1D array to a 2D array.

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

row_vector = np.array([10, 20, 30])

print("Matrix:\n", matrix)
print("Row Vector:", row_vector)

# Add row_vector to each row of the matrix
result = matrix + row_vector
print("\nMatrix + Row Vector:\n", result)
# Output:
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
```

**Explanation:**
*   `matrix` shape: `(3, 3)`
*   `row_vector` shape: `(3,)`

NumPy compares shapes from right to left:
*   Rightmost dimension: `3` (from `matrix`) and `3` (from `row_vector`) are equal. (OK)
*   Next dimension: `3` (from `matrix`) and `None` (from `row_vector` - it's a 1D array, so it's effectively `(1, 3)` for broadcasting purposes, or rather, NumPy adds a new axis of size 1 to the left).

The `row_vector` `[10, 20, 30]` is broadcast across each row of the `matrix`. It's as if `row_vector` was temporarily expanded to:
```
[[10, 20, 30],
 [10, 20, 30],
 [10, 20, 30]]
```
before the addition.

### Example 3: Column Vector Broadcasting

To broadcast a column vector, you need to explicitly make it a 2D array with a column dimension of 1.

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

col_vector = np.array([[100],
                       [200],
                       [300]])

print("\nMatrix:\n", matrix)
print("Column Vector:\n", col_vector)

# Add col_vector to each column of the matrix
result = matrix + col_vector
print("\nMatrix + Column Vector:\n", result)
# Output:
# [[101 102 103]
#  [204 205 206]
#  [307 308 309]]
```

**Explanation:**
*   `matrix` shape: `(3, 3)`
*   `col_vector` shape: `(3, 1)`

NumPy compares shapes from right to left:
*   Rightmost dimension: `3` (from `matrix`) and `1` (from `col_vector`). (OK, one is 1)
*   Next dimension: `3` (from `matrix`) and `3` (from `col_vector`). (OK, equal)

The `col_vector` `[[100], [200], [300]]` is broadcast across each column of the `matrix`. It's as if `col_vector` was temporarily expanded to:
```
[[100, 100, 100],
 [200, 200, 200],
 [300, 300, 300]]
```
before the addition.

### Example 4: Incompatible Shapes

If the broadcasting rules are not met, NumPy will raise an error.

```python
import numpy as np

arr_a = np.array([[1, 2], [3, 4]]) # Shape (2, 2)
arr_b = np.array([1, 2, 3])       # Shape (3,)

print("\nArray A:\n", arr_a)
print("Array B:", arr_b)

try:
    result = arr_a + arr_b
    print("Result:\n", result)
except ValueError as e:
    print(f"\nError: {e}")
    # Output: Error: operands could not be broadcast together with shapes (2,2) (3,)
```

**Explanation:**
*   `arr_a` shape: `(2, 2)`
*   `arr_b` shape: `(3,)`

Comparing from right to left:
*   Rightmost dimension: `2` (from `arr_a`) and `3` (from `arr_b`). Neither are equal, nor is one of them 1. So, broadcasting fails.

## Practical Applications in AI/ML

Broadcasting is incredibly useful in AI/ML for:

*   **Normalization:** Subtracting the mean and dividing by the standard deviation of a dataset.
*   **Scaling:** Multiplying data by a scalar factor.
*   **Adding Bias:** Adding a bias vector to the output of a neural network layer.
*   **Feature Engineering:** Applying a transformation to each feature column.

### Example: Data Normalization (Simple Case)

Imagine you have a dataset (a matrix) and you want to subtract the mean of each feature (column) from its respective column.

```python
import numpy as np

# Sample data: 3 samples, 2 features
data = np.array([[10, 100],
                 [20, 150],
                 [30, 120]])

print("Original Data:\n", data)

# Calculate the mean of each column (feature)
# axis=0 ensures mean is calculated down the columns
column_means = np.mean(data, axis=0)
print("Column Means:", column_means) # Output: [20. 123.33333333]

# Normalize the data by subtracting the mean of each column
# column_means (shape (2,)) is broadcast across data (shape (3, 2))
normalized_data = data - column_means
print("\nNormalized Data (mean subtracted):\n", normalized_data)
# Output:
# [[-10.         -23.33333333]
#  [  0.         26.66666667]
#  [ 10.         -3.33333333]]
```

**Explanation:** `column_means` has shape `(2,)`. When subtracted from `data` (shape `(3, 2)`), NumPy broadcasts `column_means` across the rows of `data`, effectively subtracting `[20, 123.33]` from each row.

## Assignment: Broadcasting Practice

1.  Create a 2D array `A` of shape (3, 4) with values from 1 to 12.
2.  Create a 1D array `b` of shape (4,) with values `[10, 20, 30, 40]`.
3.  Add `b` to each row of `A` using broadcasting. Print the result.
4.  Create a 2D array `C` of shape (3, 1) with values `[[1], [2], [3]]`.
5.  Multiply `A` by `C` using broadcasting. Print the result.
6.  Explain in your own words why the following operation would fail: `np.array([1, 2]) + np.array([[1], [2], [3]])`

---

In the next tutorial, we will explore Advanced Indexing techniques in NumPy, including Boolean and Fancy Indexing.

**Next:** [6. Advanced Indexing](06_advanced_indexing.md)
**Previous:** [4. Universal Functions (ufuncs)](04_universal_functions_ufuncs.md)
