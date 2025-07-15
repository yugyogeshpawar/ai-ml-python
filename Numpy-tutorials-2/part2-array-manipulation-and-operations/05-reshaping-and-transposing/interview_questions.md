# Interview Questions: Reshaping and Transposing

---

### Question 1: What is the purpose of the `-1` argument when using `ndarray.reshape()`?

**Answer:**

The `-1` is a placeholder that tells NumPy to **automatically infer the size of that dimension** based on the total number of elements in the array and the other specified dimensions. This is a convenient feature that saves you from having to manually calculate the dimension's size.

For example, if you have an array with 100 elements and you want to reshape it into a matrix with 5 columns, you can write `arr.reshape(-1, 5)`. NumPy will calculate that the first dimension must be `100 / 5 = 20`, resulting in a final shape of `(20, 5)`.

You can only use `-1` for one dimension at a time, as the rest of the dimensions must be specified for the size to be inferred correctly.

---

### Question 2: What is the difference between `ravel()` and `flatten()` for converting a multi-dimensional array to a 1D array? Which one is generally preferred and why?

**Answer:**

The key difference is that **`ravel()` returns a *view* of the original array whenever possible, while `flatten()` always returns a *copy*.**

-   **`ravel()`**: This method is more memory-efficient because it avoids creating a new copy of the data if the array is already contiguous in memory. Modifying the array returned by `ravel()` may change the original array.
-   **`flatten()`**: This method always allocates new memory and copies the data. Modifying the flattened array will never affect the original array.

**Which is preferred?**
Generally, **`ravel()` is preferred** because of its memory efficiency. In performance-critical applications, especially those involving large datasets, avoiding unnecessary data copies is crucial. You should use `flatten()` only when you have a specific need for a separate copy of the data and want to ensure the original array remains unchanged.

---

### Question 3: What does transposing an array do? If you have a 2D array `M` and you create its transpose `M_t = M.T`, is `M_t` a view or a copy? What does this imply?

**Answer:**

**Transposing** an array reverses its axes. For a 2D array, it flips the matrix over its diagonal, effectively switching the row and column indices. If the original array has a shape of `(m, n)`, its transpose will have a shape of `(n, m)`.

The transpose `M_t = M.T` is a **view**, not a copy.

**Implication:**
Because the transpose is a view, it shares the same underlying data with the original array. This means that if you modify an element in the transposed array, the change will be reflected in the original array, and vice-versa.

**Example:**
```python
M = np.array([[1, 2], [3, 4]])
M_t = M.T
M_t[0, 1] = 99 # Modifies the element at (0, 1) in the transpose

# The original M is also changed at position (1, 0)
# print(M) -> [[ 1  2]
#              [99  4]]
```
This is a memory-efficient design, but it requires the programmer to be aware that modifications to a transpose will affect the source array.
