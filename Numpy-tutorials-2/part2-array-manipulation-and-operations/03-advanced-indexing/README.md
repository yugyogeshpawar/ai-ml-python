# Part 2, Topic 3: Advanced Indexing

Beyond basic slicing, NumPy offers more sophisticated ways to select data. These advanced indexing techniques provide powerful ways to access and modify complex subsets of an array's elements. Unlike basic slicing which returns a view, **advanced indexing always returns a copy of the data**.

---

## 1. Boolean Array Indexing

This is one of the most powerful and frequently used indexing techniques. It allows you to select elements from an array based on a **boolean condition**.

The process works like this:
1.  A boolean condition is applied to the array (e.g., `arr > 5`).
2.  This produces a new boolean array of the same shape, with `True` where the condition is met and `False` otherwise.
3.  This boolean array is then used as an index, selecting only the elements from the original array that correspond to a `True` value.

**Example:**
```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 1. Create a boolean condition
condition = arr > 5
# print(condition) ->
# [[False False False]
#  [False False  True]
#  [ True  True  True]]

# 2. Use the boolean array to index the original array
print(arr[condition]) # -> [6 7 8 9]
```
The result is a 1D array containing only the elements that satisfied the condition.

You can also combine multiple conditions using the bitwise operators `&` (AND) and `|` (OR). Note: You must use `&` and `|`, not the Python keywords `and` and `or`.

```python
# Select elements greater than 2 AND less than 8
print(arr[(arr > 2) & (arr < 8)]) # -> [3 4 5 6 7]
```

Boolean indexing is essential for filtering data, a common task in data analysis.

---

## 2. Integer Array Indexing ("Fancy Indexing")

Fancy indexing allows you to select elements using an array of integer indices. This lets you pick out arbitrary elements from an array in any order.

### Indexing 1D Arrays
You can pass a list or an array of indices to select specific elements.

```python
arr_1d = np.arange(10, 20)
# -> [10 11 12 13 14 15 16 17 18 19]

# Select elements at indices 2, 5, and 8
indices = [2, 5, 8]
print(arr_1d[indices]) # -> [12 15 18]

# The order of indices matters, and you can have repeats
indices_repeat = [3, 1, 3, 9]
print(arr_1d[indices_repeat]) # -> [13 11 13 19]
```

### Indexing Multi-Dimensional Arrays
For multi-dimensional arrays, you can pass a tuple of index arrays: one for each dimension.

```python
arr_2d = np.arange(1, 10).reshape(3, 3)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Select elements at coordinates (0, 1), (1, 2), and (2, 0)
rows = np.array([0, 1, 2])
cols = np.array([1, 2, 0])
print(arr_2d[rows, cols]) # -> [2 6 7]
```
The result is a 1D array containing the elements `arr_2d[0, 1]`, `arr_2d[1, 2]`, and `arr_2d[2, 0]`.

You can also combine fancy indexing with slicing:

```python
# Select rows 0 and 2, and from those rows, select columns 1 and 2
print(arr_2d[[0, 2], 1:])
# [[2 3]
#  [8 9]]
```

Fancy indexing is incredibly flexible and allows for the selection of complex data patterns that would be difficult or inefficient to obtain with simple slicing.
