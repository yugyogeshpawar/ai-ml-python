# Exercises: Introduction to NumPy

These exercises are designed to reinforce the "why" behind NumPy and get you comfortable with the basic concepts of vectorization and performance.

---

### Exercise 1: Vectorized String Operations?

**Task:**

NumPy is optimized for *numerical* operations. But what happens if you try to perform operations on arrays of other data types, like strings?

1.  Create two Python lists of strings.
2.  Write a Python loop to concatenate the elements of these two lists element-wise (e.g., `list1[0] + list1[0]`, `list1[1] + list2[1]`, etc.).
3.  Create two NumPy arrays from the same lists of strings.
4.  Try to perform the same concatenation using the `+` operator on the NumPy arrays.
5.  Reflect on the result. Does NumPy's vectorization apply to string operations in the same way it does to numerical operations? (Hint: Look up the `numpy.char` module for vectorized string functions).

---

### Exercise 2: The Memory Footprint

**Task:**

One of the stated benefits of NumPy is its memory efficiency. Let's verify this.

1.  Import the `sys` module.
2.  Create a Python list containing 1,000,000 integers.
3.  Use `sys.getsizeof()` to check the memory size of one of the integer objects in the list and multiply it by the number of elements to get a rough idea of the total memory consumed by the data itself.
4.  Create a NumPy array containing the same 1,000,000 integers.
5.  Check the memory usage of the NumPy array using its `.nbytes` attribute.
6.  Compare the two values. Which one is more memory-efficient and why?

---

### Exercise 3: A Practical Performance Test

**Task:**

Let's simulate a simple real-world scenario: calculating the Euclidean distance between two points in a high-dimensional space. The formula for the distance between two vectors `p` and `q` is `sqrt(sum((p_i - q_i)^2))`.

1.  Define two Python lists, `p` and `q`, each containing 10,000 random numbers between 0 and 1.
2.  Write a function that calculates the Euclidean distance using a standard Python loop. Time its execution.
3.  Define two NumPy arrays, `p_np` and `q_np`, with the same data.
4.  Write a function that calculates the distance using NumPy's vectorized operations (`-`, `**`, `np.sum`, `np.sqrt`). Time its execution.
5.  Compare the performance. How does vectorization simplify both the code and its execution time?
