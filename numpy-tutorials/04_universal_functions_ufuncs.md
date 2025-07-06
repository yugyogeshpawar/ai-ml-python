# 4. Universal Functions (ufuncs)

In the previous tutorial, we covered array attributes, indexing, slicing, and basic element-wise operations. Now, we'll explore Universal Functions, or `ufuncs`, which are a core feature of NumPy that enable highly efficient operations on arrays.

## What are Ufuncs?

A Universal Function (`ufunc`) is a function that operates on `ndarray`s in an element-by-element fashion. This means that a `ufunc` takes one or more NumPy arrays as input and produces one or more NumPy arrays as output, applying the operation to each element individually.

The key advantage of `ufuncs` is that they are implemented in compiled C code, not Python. This makes them incredibly fast, especially when dealing with large arrays, as they avoid the overhead of Python loops. This "vectorization" is a cornerstone of high-performance numerical computing in Python.

## Common Mathematical Ufuncs

NumPy provides a wide range of `ufuncs` for common mathematical operations. These are applied element-wise.

### Example 1: Basic Arithmetic Ufuncs

While we saw basic arithmetic operations like `+`, `-`, `*`, `/` in the previous tutorial, these are actually implemented as `ufuncs` under the hood.

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

print("arr1:", arr1)
print("arr2:", arr2)

# Addition
print("np.add(arr1, arr2):", np.add(arr1, arr2))

# Subtraction
print("np.subtract(arr1, arr2):", np.subtract(arr1, arr2))

# Multiplication
print("np.multiply(arr1, arr2):", np.multiply(arr1, arr2))

# Division
print("np.divide(arr1, arr2):", np.divide(arr1, arr2))

# Exponentiation
print("np.power(arr1, 2):", np.power(arr1, 2))
```

### Example 2: Trigonometric Functions

```python
import numpy as np

angles_degrees = np.array([0, 30, 45, 60, 90])
angles_radians = np.deg2rad(angles_degrees) # Convert degrees to radians

print("\nAngles in Radians:", angles_radians)

# Sine
print("np.sin(angles_radians):", np.sin(angles_radians))

# Cosine
print("np.cos(angles_radians):", np.cos(angles_radians))

# Tangent
print("np.tan(angles_radians):", np.tan(angles_radians))
```

### Example 3: Exponential and Logarithmic Functions

```python
import numpy as np

arr = np.array([1, 2, 3, 4])

# Exponential (e^x)
print("\nnp.exp(arr):", np.exp(arr))

# Natural Logarithm (ln(x))
print("np.log(arr):", np.log(arr))

# Base-10 Logarithm (log10(x))
print("np.log10(arr):", np.log10(arr))
```

### Example 4: Absolute Value and Square Root

```python
import numpy as np

arr_neg = np.array([-1, -2, 3, -4])
arr_pos = np.array([1, 4, 9, 16])

# Absolute value
print("\nnp.abs(arr_neg):", np.abs(arr_neg))

# Square root
print("np.sqrt(arr_pos):", np.sqrt(arr_pos))
```

## Aggregation Functions

While not strictly `ufuncs` in the same element-wise sense, aggregation functions are closely related and also highly optimized. They perform an operation on an array and return a single value (or an array of values if an axis is specified).

### `np.sum()`: Sum of elements

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("\nOriginal Array:\n", arr)

# Sum of all elements
print("Sum of all elements:", np.sum(arr)) # Output: 21 (1+2+3+4+5+6)

# Sum along an axis
# axis=0 sums down the columns
print("Sum along axis 0 (columns):", np.sum(arr, axis=0)) # Output: [5 7 9] (1+4, 2+5, 3+6)

# axis=1 sums across the rows
print("Sum along axis 1 (rows):", np.sum(arr, axis=1)) # Output: [ 6 15] (1+2+3, 4+5+6)
```

### `np.min()`, `np.max()`: Minimum and Maximum elements

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Minimum of all elements
print("\nMinimum of all elements:", np.min(arr)) # Output: 1

# Maximum along an axis
print("Maximum along axis 0:", np.max(arr, axis=0)) # Output: [4 5 6]
```

### `np.mean()`, `np.std()`, `np.var()`: Mean, Standard Deviation, and Variance

These are crucial for statistical analysis in AI/ML.

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50])

# Mean (average)
print("\nMean:", np.mean(data)) # Output: 30.0

# Standard Deviation
print("Standard Deviation:", np.std(data)) # Output: 14.14...

# Variance
print("Variance:", np.var(data)) # Output: 200.0
```

## Assignment: Ufuncs and Aggregation

1.  Create a 1D NumPy array `x` with values from -π to π (use `np.pi` and `np.linspace` to get 100 points).
2.  Calculate the sine of each element in `x` using `np.sin()`.
3.  Calculate the exponential of each element in `x` using `np.exp()`.
4.  Create a 2D array (matrix) of shape (3, 3) with random integers between 1 and 10.
5.  Find the sum of all elements in this matrix.
6.  Find the maximum value in each row of the matrix.
7.  Calculate the mean of each column of the matrix.

---

In the next tutorial, we will explore the concept of Broadcasting, which allows NumPy to perform operations on arrays of different shapes.
