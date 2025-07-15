# Part 2, Topic 1: Universal Functions (ufuncs)

Universal Functions, or "ufuncs," are a cornerstone of NumPy's power. A ufunc is a function that operates on `ndarray` objects in an **element-by-element** fashion. This means they apply a specific operation to every single element in an array, producing a new array as output. They are the engine behind NumPy's vectorized operations.

---

## 1. What are ufuncs and Why Use Them?

At their core, ufuncs are vectorized wrappers for simple functions. For example, instead of writing a Python loop to calculate the square root of every number in an array, you can use `np.sqrt()`.

```python
import numpy as np

arr = np.array([1, 4, 9, 16])

# Without ufunc (slow Python loop)
result_loop = []
for x in arr:
    result_loop.append(np.sqrt(x)) # Still calls numpy, but in a loop

# With ufunc (fast and vectorized)
result_ufunc = np.sqrt(arr)

# print(result_ufunc) -> [1. 2. 3. 4.]
```
The ufunc version is not just cleaner to write; it's significantly faster because the looping happens in compiled C code, not in Python.

---

## 2. Types of ufuncs

Ufuncs can be broadly categorized into two types:

### Unary ufuncs
These functions operate on a single array.

**Common Unary ufuncs:**

| Function      | Description                               |
|---------------|-------------------------------------------|
| `np.sqrt`     | Compute the non-negative square-root.     |
| `np.exp`      | Calculate the exponential of all elements.|
| `np.log`, `np.log10` | Natural logarithm and base-10 logarithm. |
| `np.sin`, `np.cos`, `np.tan` | Standard trigonometric functions. |
| `np.abs`      | Compute the absolute value.               |
| `np.square`   | Compute the square of each element.       |
| `np.ceil`, `np.floor` | Round up or down to the nearest integer. |
| `np.isnan`    | Returns a boolean array indicating if a value is `NaN` (Not a Number). |
| `np.isfinite`, `np.isinf` | Returns a boolean array indicating if a value is finite or infinite. |

**Example:**
```python
arr = np.array([-2, -1, 0, 1, 2])
print(np.abs(arr)) # -> [2 1 0 1 2]

angles = np.array([0, np.pi/2, np.pi])
print(np.sin(angles)) # -> [0. 1. 1.2246468e-16] (Note the floating point inaccuracy for sin(pi))
```

### Binary ufuncs
These functions operate on two arrays, performing an element-wise combination. The arrays must be compatible for broadcasting (which we'll cover in the next lesson).

**Common Binary ufuncs:**

| Function      | Description                               | Operator |
|---------------|-------------------------------------------|----------|
| `np.add`      | Element-wise addition.                    | `+`      |
| `np.subtract` | Element-wise subtraction.                 | `-`      |
| `np.multiply` | Element-wise multiplication.              | `*`      |
| `np.divide`   | Element-wise division.                    | `/`      |
| `np.power`    | Element-wise power.                       | `**`     |
| `np.maximum`, `np.minimum` | Element-wise maximum or minimum of two arrays. | |
| `np.mod`      | Element-wise remainder of division.       | `%`      |
| `np.greater`, `np.less` | Element-wise comparison, returns a boolean array. | `>`, `<` |

**Example:**
```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(np.add(a, b))      # -> [11 22 33 44]
print(a * b)             # Same as np.multiply(a, b) -> [10 40 90 160]

# Find which elements in 'a' are greater than elements in a new array
c = np.array([0, 3, 2, 5])
print(np.greater(a, c))  # -> [ True False  True False]
```

Ufuncs are the building blocks of almost all numerical computations in NumPy. Understanding how to use them effectively is key to writing clean, fast, and powerful data analysis code.
