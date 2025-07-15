# Part 1, Topic 3: Array Attributes and Data Types

Once you have a NumPy array, you can inspect its properties to understand its structure and the nature of the data it holds. These properties are stored as attributes of the `ndarray` object.

---

## 1. Essential Array Attributes

These attributes provide metadata about the array. They do not return a new array but give you information about the existing one.

Let's use this 2D array as our example:
```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
```

### `ndarray.ndim`
Returns the number of dimensions (or axes) of the array. For a 2D array, this is 2.

```python
print(arr.ndim)  # -> 2
```
- A 1D array has `ndim = 1`.
- A 3D array has `ndim = 3`.

### `ndarray.shape`
Returns a tuple of integers indicating the size of the array in each dimension.

```python
print(arr.shape)  # -> (2, 3)
```
This means the array has 2 rows and 3 columns. For a 1D array of 5 elements, the shape would be `(5,)`.

### `ndarray.size`
Returns the total number of elements in the array. This is the product of the elements of the shape tuple.

```python
print(arr.size)  # -> 6 (because 2 * 3 = 6)
```

---

## 2. Array Data Types (`dtype`)

Unlike Python lists, all elements in a NumPy array must be of the same data type. The **data type** or `dtype` is a special object that provides information about the type of data (e.g., integer, float, boolean) and how much memory it occupies.

### `ndarray.dtype`
You can access the data type of an array using the `.dtype` attribute.

```python
# An array of integers
int_arr = np.array([1, 2, 3])
print(int_arr.dtype)  # -> int64 (on most systems)

# An array containing a float
float_arr = np.array([1.0, 2, 3])
print(float_arr.dtype)  # -> float64
```
NumPy automatically infers the most appropriate data type. If you mix integers and floats, it will upcast the integers to floats to maintain homogeneity.

### Specifying Data Types on Creation
You can explicitly specify the data type when you create an array using the `dtype` argument. This is useful for controlling memory usage and ensuring precision.

```python
# Create an array of floats, even with integer inputs
float_array = np.array([1, 2, 3], dtype=np.float32)
print(float_array.dtype)  # -> float32

# Create an array of 8-bit integers (range -128 to 127)
small_int_array = np.zeros((2, 3), dtype=np.int8)
print(small_int_array.dtype) # -> int8
```

### Common NumPy Data Types

| Data Type        | Description                               |
| ---------------- | ----------------------------------------- |
| `np.int8`, `np.int16`, `np.int32`, `np.int64` | Signed integers of different sizes. |
| `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`| Unsigned integers (non-negative). |
| `np.float16`, `np.float32`, `np.float64` | Floating-point numbers with different precision. `np.float64` is the default and corresponds to Python's `float`. |
| `np.complex64`, `np.complex128` | Complex numbers. |
| `np.bool`        | Boolean type storing `True` and `False` values. |
| `np.object`      | For storing Python objects (use with caution). |
| `np.string_`     | Fixed-length string type. |
| `np.unicode_`    | Fixed-length unicode type. |

Choosing the right `dtype` can significantly reduce the memory footprint of your arrays, which is critical when working with very large datasets.
