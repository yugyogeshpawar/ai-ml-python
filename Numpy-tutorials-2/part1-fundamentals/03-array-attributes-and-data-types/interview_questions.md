# Interview Questions: Array Attributes and Data Types

---

### Question 1: What is the difference between the `size`, `shape`, and `ndim` attributes of a NumPy array?

**Answer:**

These three attributes describe the structure of a NumPy array in different ways:

-   **`ndim` (Number of Dimensions):** This is an integer that represents the number of axes, or the "rank," of the array.
    -   A vector like `[1, 2, 3]` has `ndim = 1`.
    -   A matrix like `[[1, 2], [3, 4]]` has `ndim = 2`.
    -   A 3D tensor would have `ndim = 3`.

-   **`shape`:** This is a **tuple** of integers where each integer represents the number of elements along that axis (dimension). The length of the `shape` tuple is equal to `ndim`.
    -   For a vector `[1, 2, 3]`, `shape` is `(3,)`.
    -   For a 2x3 matrix `[[1, 2, 3], [4, 5, 6]]`, `shape` is `(2, 3)`.

-   **`size`:** This is a single integer representing the **total number of elements** in the array. It is always equal to the product of the elements of the `shape` tuple.
    -   For a 2x3 matrix, `size` is `2 * 3 = 6`.

In summary: `ndim` tells you how many numbers are in the `shape` tuple, `shape` tells you the length of each dimension, and `size` tells you the total count of elements.

---

### Question 2: Why is choosing the correct `dtype` in NumPy important, especially when working with large datasets?

**Answer:**

Choosing the correct `dtype` is crucial for two main reasons: **memory efficiency** and **data integrity**.

1.  **Memory Efficiency:** NumPy arrays store data in a contiguous block of memory. The size of this block is determined by `array.size * array.itemsize`. The `itemsize` is directly dependent on the `dtype`. For example, an `int64` requires 8 bytes per element, while a `uint8` requires only 1 byte. If you have a large dataset of a million numbers that are known to be between 0 and 255, storing them as `int64` would use 8 MB of memory, whereas storing them as `uint8` would use only 1 MB. This 8x memory saving is critical in big data applications where memory can be a bottleneck.

2.  **Data Integrity:** Using an inappropriate `dtype` can lead to silent errors.
    -   **Overflow:** If you try to store a value larger than the `dtype`'s maximum (e.g., storing `300` in a `uint8` which has a max of `255`), the value will "wrap around" (overflow), resulting in an incorrect value (e.g., `300` might become `44`). This happens without raising an error and can corrupt your data and calculations.
    -   **Truncation:** When casting from a float to an integer (e.g., using `.astype(np.int32)`), the decimal part is truncated, not rounded. For example, `3.99` becomes `3`. This loss of precision can be a problem if not intended.

Therefore, selecting the smallest `dtype` that can safely represent the full range of your data is a best practice in numerical computing.

---

### Question 3: What happens when you create a NumPy array with a mix of integer and floating-point numbers? Explain the concept of "upcasting."

**Answer:**

When you create a NumPy array from a collection of numbers containing both integers and floating-point values, NumPy will **upcast** all the elements to a more general type to maintain the rule that all elements in an array must have the same `dtype`.

**Upcasting** is the process of converting data from a less precise or less general data type to a more precise or more general one. In this case, floating-point numbers are more general than integers because they can represent fractional values.

For example:
`np.array([1, 2, 3.14, 4])`

1.  NumPy scans the elements and finds integers (`1`, `2`, `4`) and a float (`3.14`).
2.  To ensure all elements can be stored under a single `dtype`, it identifies the most general type, which is `float64` (the default float type).
3.  It then converts all the integers (`1`, `2`, `4`) into their floating-point equivalents (`1.0`, `2.0`, `4.0`).
4.  The final array will be `array([1.  , 2.  , 3.14, 4.  ])` with a `dtype` of `float64`.

This process ensures type homogeneity at the cost of using more memory, as floats typically require more storage than integers. This is a safe default behavior because it prevents data loss (you can represent any integer as a float, but not vice-versa without losing precision).
