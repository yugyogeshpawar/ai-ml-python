# Exercises: Array Attributes and Data Types

These exercises will help you get hands-on experience with inspecting array properties and managing data types.

---

### Exercise 1: The Anatomy of an Array

**Task:**

1.  Create a 3D NumPy array with a shape of `(4, 3, 2)`. You can fill it with any numbers you like. A quick way is to use `np.arange()` and then `.reshape()`. For example: `np.arange(24).reshape((4, 3, 2))`.
2.  Print the array itself.
3.  On separate lines, print the following attributes of the array:
    *   Number of dimensions (`.ndim`)
    *   Shape (`.shape`)
    *   Size (`.size`)
    *   Data type (`.dtype`)
    *   The size in bytes of each element in the array (`.itemsize`)
    *   The total bytes consumed by the elements of the array (`.nbytes`)
4.  Add a print statement that verifies that `size * itemsize` is equal to `nbytes`.

---

### Exercise 2: Memory Optimization with `dtype`

**Task:**

Imagine you are working with a dataset of student test scores, which are always integers between 0 and 100.

1.  First, create a large NumPy array (e.g., 1,000,000 elements) of random integers between 0 and 100. Let NumPy use its default `dtype` (`int64`). You can use `np.random.randint(0, 101, 1000000)`.
2.  Print the data type (`.dtype`) and the total memory consumption in megabytes (`.nbytes / 1024 / 1024`).
3.  Now, think about the most memory-efficient, yet safe, `dtype` for storing numbers between 0 and 100. (Hint: Look at the `np.uint8` type, which can store integers from 0 to 255).
4.  Create a new array with the same data, but this time, explicitly set the `dtype` to your chosen efficient type.
5.  Print the new array's data type and its memory consumption in megabytes.
6.  Calculate and print the percentage of memory you saved by choosing a more appropriate data type.

---

### Exercise 3: Data Type Casting and Its Consequences

**Task:**

This exercise demonstrates what happens when you change an array's data type after it has been created.

1.  Create a NumPy array containing floating-point numbers, for example: `float_arr = np.array([1.1, 2.7, 3.5, 4.9])`.
2.  Print the original array and its `dtype`.
3.  Now, create a new array by casting the `float_arr` to an integer type using the `.astype()` method. For example: `int_arr = float_arr.astype(np.int32)`.
4.  Print the new integer array and its `dtype`.
5.  Observe what happened to the decimal values. Write a comment in your script explaining the casting behavior (e.g., does it round to the nearest integer, or does it truncate?).
6.  Now, create a boolean array by casting the `float_arr` to `np.bool`. What value(s) cast to `False` and what value(s) cast to `True`? Test this by adding `0.0` to your original `float_arr` and re-running the cast.
