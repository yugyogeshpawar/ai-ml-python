# Exercises: Universal Functions (ufuncs)

These exercises will help you practice using ufuncs to perform efficient, element-wise operations.

---

### Exercise 1: The Sigmoid Function

**Task:**

In machine learning, the sigmoid function is a common activation function. It maps any value to a value between 0 and 1. The formula is: `S(x) = 1 / (1 + e^(-x))`.

1.  Create a 1D NumPy array named `x` containing a range of values from -10 to 10. `np.linspace(-10, 10, 21)` is a good choice.
2.  Write a Python function `sigmoid(x)` that takes a NumPy array `x` as input.
3.  Inside the function, use NumPy's ufuncs (`np.exp`, `+`, `/`) to implement the sigmoid formula in a single, vectorized line of code.
4.  Call your function with the array `x` and print the result.
5.  Verify that the values are all between 0 and 1.

---

### Exercise 2: Clipping and Replacing Values

**Task:**

Imagine you have a dataset of sensor readings that are noisy. You want to "clip" the data so that any value below a certain minimum is set to that minimum, and any value above a maximum is set to that maximum.

1.  Create a 1D NumPy array of 20 random integers between 0 and 100.
2.  Define a `min_threshold` of 20 and a `max_threshold` of 80.
3.  Use the `np.maximum` and `np.minimum` ufuncs to clip the array.
    -   First, use `np.maximum` to replace all values below `min_threshold` with `min_threshold`.
    -   Then, on the result of the first step, use `np.minimum` to replace all values above `max_threshold` with `max_threshold`.
4.  Print the original array and the final, clipped array.
5.  (Bonus) NumPy has a built-in function `np.clip` that does this. Try to achieve the same result using `np.clip` in a single line.

---

### Exercise 3: In-Place Operations for Memory Efficiency

**Task:**

This exercise demonstrates the use of the `out` argument to perform operations without creating new arrays.

1.  Create two large 1D arrays, `a` and `b`, each with 1,000,000 random numbers.
2.  Create a third array, `c`, of the same size, initialized to all zeros using `np.zeros()`.
3.  Perform the operation `a * b + a` and store the result in a new variable `result_new`.
4.  Now, perform the same operation, but do it in-place using the `out` argument to save memory.
    -   First, use `np.multiply(a, b, out=c)` to store the product in `c`.
    -   Next, use `np.add(c, a, out=c)` to add `a` to `c`, overwriting `c` with the final result.
5.  Use `np.array_equal()` to verify that `result_new` and `c` contain the same values. This shows that you can achieve the same result while reusing memory.
