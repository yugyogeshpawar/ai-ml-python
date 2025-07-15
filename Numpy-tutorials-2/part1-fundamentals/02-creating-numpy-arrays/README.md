# Part 1, Topic 2: Creating NumPy Arrays

Now that we understand *why* NumPy is essential, let's learn how to create its core object: the `ndarray`. There are several ways to create NumPy arrays, each suited for different situations.

---

## 1. Creating Arrays from Existing Data (e.g., Python Lists)

The most common way to create an array is by converting a Python list or tuple using the `np.array()` function.

### One-Dimensional Array
You can create a simple 1D array from a list:

```python
import numpy as np

my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)
# print(my_array) -> [1 2 3 4 5]
```

### Multi-Dimensional Array
To create a 2D array (or matrix), you pass a list of lists:

```python
my_2d_list = [[1, 2, 3], [4, 5, 6]]
my_2d_array = np.array(my_2d_list)
# print(my_2d_array) ->
# [[1 2 3]
#  [4 5 6]]
```
NumPy will automatically infer the data type of the array elements (e.g., integers, floats). We'll cover data types in the next lesson.

---

## 2. Creating Arrays from Scratch: Placeholder Functions

Often, you need to create an array of a specific size without knowing the exact values in advance. NumPy provides several functions for this purpose.

### `np.zeros()`
Creates an array filled with zeros. You pass a tuple representing the desired shape of the array.

```python
# A 1D array of 5 zeros
zeros_1d = np.zeros(5) # -> [0. 0. 0. 0. 0.]

# A 2D array (3 rows, 4 columns) of zeros
zeros_2d = np.zeros((3, 4))
# -> [[0. 0. 0. 0.]
#     [0. 0. 0. 0.]
#     [0. 0. 0. 0.]]
```
*Note: By default, the data type is `float64`.*

### `np.ones()`
Similar to `np.zeros()`, but creates an array filled with ones.

```python
# A 1D array of 3 ones
ones_1d = np.ones(3) # -> [1. 1. 1.]

# A 3D array (2 layers, 3 rows, 2 columns) of ones
ones_3d = np.ones((2, 3, 2))
# -> [[[1. 1.]
#      [1. 1.]
#      [1. 1.]]
#
#     [[1. 1.]
#      [1. 1.]
#      [1. 1.]]]
```

---

## 3. Creating Arrays with Specific Sequences

NumPy also makes it easy to create arrays with regular sequences of numbers.

### `np.arange()`
This function is similar to Python's built-in `range()`, but it returns a NumPy array instead of a list iterator. It supports integer and floating-point steps.

**Syntax:** `np.arange(start, stop, step)`

```python
# An array from 0 up to (but not including) 10
range_array = np.arange(10) # -> [0 1 2 3 4 5 6 7 8 9]

# An array from 2 to 8, with a step of 2
step_array = np.arange(2, 9, 2) # -> [2 4 6 8]
```

### `np.linspace()`
Creates an array with a specified number of points, evenly spaced between a start and end value. This is particularly useful for plotting or when you need a specific number of samples in a given range.

**Syntax:** `np.linspace(start, stop, num_points)`

```python
# Create 5 evenly spaced points between 0 and 1 (inclusive)
linspace_array = np.linspace(0, 1, 5)
# -> [0.   0.25 0.5  0.75 1.  ]

# Create 10 points from 0 to 50
data_points = np.linspace(0, 50, 10)
# -> [ 0.          5.55555556 11.11111111 16.66666667 22.22222222
#     27.77777778 33.33333333 38.88888889 44.44444444 50.        ]
```

These functions are the fundamental building blocks for nearly all work done in NumPy. Mastering them is the first step toward effective numerical computing in Python.
