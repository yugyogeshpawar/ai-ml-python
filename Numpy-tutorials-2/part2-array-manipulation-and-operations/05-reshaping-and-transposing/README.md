# Part 2, Topic 5: Reshaping and Transposing

Changing the shape and orientation of an array without changing its data is a common and critical task. NumPy provides a powerful set of tools for rearranging the layout of array elements.

---

## 1. Reshaping Arrays

The most flexible way to change an array's shape is with the `reshape()` method. This allows you to rearrange the elements of an array into a new shape, as long as the new shape is compatible with the original size.

**Key Rule:** The `size` of the reshaped array must be equal to the `size` of the original array.

```python
import numpy as np

arr = np.arange(1, 13) # size = 12
# -> [ 1  2  3  4  5  6  7  8  9 10 11 12]

# Reshape to a 3x4 matrix
reshaped_arr = arr.reshape(3, 4)
# print(reshaped_arr) ->
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# Reshape to a 4x3 matrix
reshaped_arr_2 = arr.reshape(4, 3)
# print(reshaped_arr_2) ->
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
```

### The `-1` Inference
You can use `-1` as a placeholder for one of the dimensions, and NumPy will automatically infer the correct size for that dimension based on the array's total size.

```python
# Reshape to have 2 rows; NumPy calculates the number of columns
arr.reshape(2, -1) # -> Shape becomes (2, 6)

# Reshape to have 4 columns; NumPy calculates the number of rows
arr.reshape(-1, 4) # -> Shape becomes (3, 4)
```
This is extremely useful when you don't want to manually calculate the dimension size.

---

## 2. Flattening Arrays: `ravel()` and `flatten()`

Sometimes you need to convert a multi-dimensional array into a 1D array. This is known as "flattening."

-   **`ravel()`**: Returns a **view** of the original array whenever possible. This is memory-efficient as it doesn't create a new copy of the data unless it has to (e.g., if the array's data is not contiguous in memory).
-   **`flatten()`**: Always returns a **copy** of the data.

```python
arr_2d = np.array([[1, 2], [3, 4]])

raveled = arr_2d.ravel()
flattened = arr_2d.flatten()

# Modifying the output of ravel() will affect the original array
raveled[0] = 100
# print(arr_2d) -> [[100, 2], [3, 4]]

# Modifying the output of flatten() will NOT affect the original
flattened[1] = 200
# arr_2d is unchanged by this operation
```
**Best Practice:** Use `ravel()` unless you have a specific reason to require a copy of the data.

---

## 3. Transposing Arrays

Transposing is a special form of reshaping that flips an array over its diagonal. It switches the row and column indices. The most common way to do this is with the `.T` attribute.

```python
arr = np.arange(1, 7).reshape(2, 3)
# [[1 2 3]
#  [4 5 6]]

# Transpose the array
transposed_arr = arr.T
# print(transposed_arr) ->
# [[1 4]
#  [2 5]
#  [3 6]]

print(arr.shape)          # -> (2, 3)
print(transposed_arr.shape) # -> (3, 2)
```
Like `ravel()`, the `.T` attribute returns a **view** of the original data, not a copy. Modifying the transposed array will modify the original.

Transposing is fundamental in linear algebra, for example, when computing a dot product between a matrix and its transpose.
