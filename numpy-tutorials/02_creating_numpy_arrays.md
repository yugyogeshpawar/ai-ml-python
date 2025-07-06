# 2. Creating NumPy Arrays

In the previous tutorial, we learned what NumPy is and why it's essential for AI/ML. Now, let's dive into the practical aspects of NumPy by understanding how to create NumPy arrays (ndarrays) in various ways.

## 2.1 Creating Arrays from Python Lists

The most straightforward way to create a NumPy array is by converting a standard Python list or a nested list (for multi-dimensional arrays).

### Example 1: Creating a 1D Array

```python
import numpy as np

# Create a 1D array from a Python list
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)

print("1D Array:")
print(array_1d)
print("Type:", type(array_1d))
print("Shape:", array_1d.shape)
```

**Explanation:**
*   `np.array()` is the function used to create an array.
*   `type(array_1d)` confirms it's a `numpy.ndarray`.
*   `array_1d.shape` shows the dimensions of the array. For a 1D array with 5 elements, the shape is `(5,)`.

### Example 2: Creating a 2D Array (Matrix)

```python
import numpy as np

# Create a 2D array from a nested Python list
list_2d = [[1, 2, 3], [4, 5, 6]]
array_2d = np.array(list_2d)

print("\n2D Array:")
print(array_2d)
print("Shape:", array_2d.shape)
print("Number of dimensions (ndim):", array_2d.ndim)
```

**Explanation:**
*   For a 2D array (matrix), the shape `(2, 3)` means 2 rows and 3 columns.
*   `array_2d.ndim` confirms it's a 2-dimensional array.

## 2.2 Intrinsic Array Creation Functions

NumPy provides several built-in functions to create arrays with initial placeholder content, which are very useful when you know the size and shape of the array you need but don't have the exact data yet.

### `np.zeros()`: Create an array filled with zeros

```python
import numpy as np

# Create a 1D array of 5 zeros
zeros_1d = np.zeros(5)
print("\nArray of Zeros (1D):")
print(zeros_1d)

# Create a 2D array (3 rows, 4 columns) filled with zeros
zeros_2d = np.zeros((3, 4))
print("\nArray of Zeros (2D):")
print(zeros_2d)
```

### `np.ones()`: Create an array filled with ones

```python
import numpy as np

# Create a 1D array of 3 ones
ones_1d = np.ones(3)
print("\nArray of Ones (1D):")
print(ones_1d)

# Create a 2D array (2 rows, 5 columns) filled with ones
ones_2d = np.ones((2, 5))
print("\nArray of Ones (2D):")
print(ones_2d)
```

### `np.full()`: Create an array filled with a specific value

```python
import numpy as np

# Create a 2x2 array filled with the number 7
full_array = np.full((2, 2), 7)
print("\nArray filled with 7:")
print(full_array)
```

### `np.arange()`: Create arrays with a range of values

Similar to Python's `range()`, but returns a NumPy array.

```python
import numpy as np

# Create an array with values from 0 to 9 (exclusive of 10)
range_array_1 = np.arange(10)
print("\nArray with range (0-9):")
print(range_array_1)

# Create an array with values from 5 to 15 (exclusive of 16)
range_array_2 = np.arange(5, 16)
print("\nArray with range (5-15):")
print(range_array_2)

# Create an array with values from 0 to 10 with a step of 2
range_array_3 = np.arange(0, 11, 2)
print("\nArray with range and step (0-10, step 2):")
print(range_array_3)
```

### `np.linspace()`: Create arrays with evenly spaced values

This function is useful when you need a specific number of points evenly distributed over a given interval.

```python
import numpy as np

# Create an array of 5 evenly spaced values between 0 and 1 (inclusive)
linspace_array = np.linspace(0, 1, 5)
print("\nArray with 5 evenly spaced values between 0 and 1:")
print(linspace_array)
```

## 2.3 Creating Random Arrays

Random arrays are frequently used in AI/ML for tasks like initializing neural network weights, generating synthetic data, or sampling. NumPy's `random` module provides various functions for this.

### `np.random.rand()`: Create an array of random numbers between 0 and 1

```python
import numpy as np

# Create a 1D array of 3 random numbers
rand_1d = np.random.rand(3)
print("\nRandom 1D Array (0-1):")
print(rand_1d)

# Create a 2x3 array of random numbers
rand_2d = np.random.rand(2, 3)
print("\nRandom 2D Array (0-1):")
print(rand_2d)
```

### `np.random.randn()`: Create an array of random numbers from a standard normal distribution

This generates numbers from a "standard normal" (Gaussian) distribution with mean 0 and variance 1.

```python
import numpy as np

# Create a 1D array of 3 random numbers from standard normal distribution
randn_1d = np.random.randn(3)
print("\nRandom 1D Array (Standard Normal):")
print(randn_1d)

# Create a 2x3 array of random numbers from standard normal distribution
randn_2d = np.random.randn(2, 3)
print("\nRandom 2D Array (Standard Normal):")
print(randn_2d)
```

### `np.random.randint()`: Create an array of random integers

```python
import numpy as np

# Create a 1D array of 5 random integers between 0 (inclusive) and 10 (exclusive)
randint_1d = np.random.randint(0, 10, 5)
print("\nRandom Integers (0-9, 5 numbers):")
print(randint_1d)

# Create a 2x2 array of random integers between 10 (inclusive) and 20 (exclusive)
randint_2d = np.random.randint(10, 20, size=(2, 2))
print("\nRandom Integers (10-19, 2x2 array):")
print(randint_2d)
```

## Assignment: Array Creation Practice

1.  Create a 3x3 NumPy array filled with the number 9.
2.  Generate a 1D array containing 7 evenly spaced numbers between 10 and 20 (inclusive).
3.  Create a 4x2 array of random floating-point numbers between 0 and 1.
4.  Create a 1D array of integers from 100 down to 90 (inclusive).

---

In the next tutorial, we will explore array attributes and basic operations like indexing and slicing.

**Next:** [3. Array Attributes and Basics](03_array_attributes_and_basics.md)
**Previous:** [1. Introduction to NumPy](01_introduction_to_numpy.md)
