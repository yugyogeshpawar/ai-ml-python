# Part 1, Topic 4: Basic Indexing and Slicing

Accessing and modifying subsets of data is a fundamental part of data analysis. NumPy provides a powerful and intuitive syntax for indexing and slicing arrays, which is similar to Python's list indexing but with extended capabilities for multiple dimensions.

---

## 1. Indexing in 1D Arrays

Indexing for one-dimensional arrays works just like it does for Python lists. You use square brackets `[]` to access elements by their position, starting from index 0.

```python
import numpy as np

arr_1d = np.arange(10, 20)
# -> [10 11 12 13 14 15 16 17 18 19]

# Get the first element
print(arr_1d[0])  # -> 10

# Get the third element
print(arr_1d[2])  # -> 12

# Get the last element using negative indexing
print(arr_1d[-1]) # -> 19

# You can also modify elements in place
arr_1d[0] = 100
# print(arr_1d) -> [100  11  12  13  14  15  16  17  18  19]
```

---

## 2. Slicing in 1D Arrays

Slicing allows you to select a range of elements. The syntax is `start:stop:step`, where `stop` is exclusive.

```python
# Slice elements from index 2 up to (but not including) index 5
print(arr_1d[2:5])  # -> [12 13 14]

# Slice from the beginning up to index 4
print(arr_1d[:4])   # -> [100 11 12 13]

# Slice from index 5 to the end of the array
print(arr_1d[5:])   # -> [15 16 17 18 19]

# Get every other element from the entire array
print(arr_1d[::2])  # -> [100 12 14 16 18]
```

### An Important Note: Slices are Views, Not Copies!

A key feature (and a common pitfall for beginners) is that **NumPy slices are views into the original array**. This means the slice is not a new array; it's a pointer to a section of the original array's data. Modifying a slice will modify the original array.

```python
arr_1d = np.arange(5) # -> [0 1 2 3 4]
my_slice = arr_1d[2:4] # -> [2 3]

my_slice[0] = 99 # Modify the first element of the slice

print(my_slice)    # -> [99  3]
print(arr_1d)      # -> [ 0  1 99  3  4] <- The original array is changed!
```
This behavior is designed for performance and memory efficiency. If you need a true copy, you must use the `.copy()` method: `my_slice_copy = arr_1d[2:4].copy()`.

---

## 3. Indexing and Slicing in 2D Arrays (Matrices)

For multi-dimensional arrays, you provide an index or slice for each dimension, separated by commas. The syntax is `array[row_index, column_index]`.

Let's use this 2D array:
```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
```

### Indexing Single Elements
```python
# Get the element at row 1, column 2
print(arr_2d[1, 2]) # -> 6
```

### Slicing Sub-Matrices
You can combine integer indexing and slicing to select rows, columns, or sub-regions.

```python
# Get the first two rows
# : for rows means "all rows", but we specify up to 2
print(arr_2d[:2, :])
# [[1 2 3]
#  [4 5 6]]

# Get the second column
# : for rows means "all rows", 1 for columns means "only index 1"
print(arr_2d[:, 1]) # -> [2 5 8] (This is a 1D array)

# Get a 2x2 sub-matrix from the top-right
# Rows 0-1, Columns 1-2
print(arr_2d[:2, 1:])
# [[2 3]
#  [5 6]]
```

This powerful indexing is fundamental to virtually all data manipulation tasks in NumPy and the libraries that depend on it.
