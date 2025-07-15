# Exercises: Reshaping and Transposing

These exercises will help you practice rearranging the dimensions of your arrays.

---

### Exercise 1: Image Batch Reshaping

**Task:**

Imagine you have a batch of 10 grayscale images, each with a resolution of 28x28 pixels. For processing in a machine learning model, you need to flatten each image into a 1D vector of 784 pixels (`28 * 28 = 784`).

1.  Create a random 3D NumPy array of shape `(10, 28, 28)` to simulate the batch of images.
2.  Use the `reshape()` method to convert this 3D array into a 2D array where each *row* represents a flattened image.
3.  What should the final shape of the array be? Use the `-1` inference trick in your reshape call to have NumPy calculate one of the dimensions for you.
4.  Print the shape of the original array and the shape of the reshaped array to confirm you did it correctly.

---

### Exercise 2: `ravel()` vs. `flatten()` in Practice

**Task:**

This exercise will highlight the practical difference between `ravel()` (view) and `flatten()` (copy).

1.  Create a 4x4 matrix with integers from 0 to 15.
2.  Create two new variables, `raveled_view` and `flattened_copy`, by applying the respective methods to your original matrix.
3.  Write a function called `modify_array(arr)` that takes a 1D array as input and sets its first element to `99`.
4.  Call this function once with `raveled_view` and once with `flattened_copy`.
5.  Print your original 4x4 matrix after calling the function on both arrays.
6.  Write a comment explaining which function call modified the original matrix and why.

---

### Exercise 3: Matrix-Vector Multiplication with Transposition

**Task:**

In linear algebra, to multiply a matrix `M` by a vector `v`, the inner dimensions must match. If you have a matrix of shape `(m, n)` and a vector of shape `(n,)`, `M @ v` works. But what if your vector has the same number of elements as the rows of `M`?

1.  Create a matrix `M` of shape `(4, 3)` with random integers.
2.  Create a vector `v` of shape `(4,)` with random integers.
3.  Try to compute the dot product `M @ v`. It will fail. Why?
4.  To make the multiplication work, you can multiply the *transpose* of `M` by `v`.
5.  Calculate `M.T @ v`.
6.  Print the shapes of `M`, `v`, and `M.T` and write a comment explaining why `M.T @ v` is a valid operation while `M @ v` is not.
