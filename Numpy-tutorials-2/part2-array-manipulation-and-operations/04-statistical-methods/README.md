# Part 2, Topic 4: Statistical Methods

A primary use case for NumPy is performing statistical analysis on datasets. NumPy provides a suite of fast, vectorized aggregate functions that can compute summary statistics like the mean, standard deviation, and sum. These functions can be called either as methods of the `ndarray` object (e.g., `arr.sum()`) or as top-level NumPy functions (e.g., `np.sum(arr)`).

---

## 1. Basic Aggregate Functions

These functions take an array and return a single value that summarizes the data.

Let's consider a simple array:
```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
```

**Common Aggregate Functions:**

| Method Syntax | Function Syntax | Description |
|---------------|-----------------|-------------|
| `arr.sum()`   | `np.sum(arr)`   | Sum of all elements. |
| `arr.mean()`  | `np.mean(arr)`  | Mean (average) of all elements. |
| `arr.std()`   | `np.std(arr)`   | Standard deviation. |
| `arr.var()`   | `np.var(arr)`   | Variance. |
| `arr.min()`   | `np.min(arr)`   | Minimum value. |
| `arr.max()`   | `np.max(arr)`   | Maximum value. |
| `arr.argmin()`| `np.argmin(arr)`| Index of the minimum value. |
| `arr.argmax()`| `np.argmax(arr)`| Index of the maximum value. |
| `arr.cumsum()`| `np.cumsum(arr)`| Cumulative sum of elements. |
| `arr.cumprod()`|`np.cumprod(arr)`| Cumulative product of elements. |

**Example:**
```python
print(f"Sum: {arr.sum()}")         # -> 15
print(f"Mean: {arr.mean()}")        # -> 3.0
print(f"Max value: {arr.max()}")    # -> 5
print(f"Index of max value: {arr.argmax()}") # -> 4
print(f"Cumulative sum: {arr.cumsum()}") # -> [ 1  3  6 10 15]
```

---

## 2. Aggregations Along Axes

The real power of these functions becomes apparent when working with multi-dimensional arrays. You can compute statistics along a specific **axis** (dimension). The `axis` argument tells NumPy which dimension to "collapse" during the calculation.

Consider this 2D array:
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
```

### `axis=0` (Collapse along rows)
This computes the statistic for each **column**.

```python
# Calculate the sum of each column
col_sums = arr_2d.sum(axis=0)
print(col_sums) # -> [12 15 18]
# Calculation: [1+4+7, 2+5+8, 3+6+9]
```
The result is a 1D array where each element is the sum of the corresponding column.

### `axis=1` (Collapse along columns)
This computes the statistic for each **row**.

```python
# Calculate the mean of each row
row_means = arr_2d.mean(axis=1)
print(row_means) # -> [2. 6. 8.]
# Calculation: [mean(1,2,3), mean(4,5,6), mean(7,8,9)]
```
The result is a 1D array where each element is the mean of the corresponding row.

This ability to aggregate along axes is fundamental to data analysis, allowing you to quickly summarize complex datasets by feature (columns) or by sample (rows).

---

## 3. Boolean Arrays and Statistical Methods

Statistical methods also work on boolean arrays. `True` is treated as `1` and `False` is treated as `0`. This provides a convenient way to count the number of `True` values.

```python
bools = np.array([True, False, True, True, False])

# Count the number of True values
print(bools.sum()) # -> 3

# Check if any value is True
print(bools.any()) # -> True

# Check if all values are True
print(bools.all()) # -> False
```
This is often used with boolean indexing. For example, you can count how many elements in an array satisfy a condition:
```python
arr = np.random.randn(100)
num_positive = (arr > 0).sum()
print(f"Number of positive values: {num_positive}")
