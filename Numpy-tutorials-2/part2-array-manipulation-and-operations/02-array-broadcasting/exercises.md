# Exercises: Array Broadcasting

These exercises will help you master the rules of broadcasting to perform operations on arrays of different shapes.

---

### Exercise 1: Normalizing Data

**Task:**

A common task in data science is to "normalize" data by subtracting the mean and dividing by the standard deviation for each feature. This centers the data around zero.

1.  Create a 5x3 NumPy array of random numbers. This represents a dataset with 5 samples and 3 features.
2.  Calculate the mean of each *column* (feature). The result should be a 1D array of shape `(3,)`. (Hint: use the `axis` argument in `np.mean()`).
3.  Subtract the mean vector from the original data matrix. Use broadcasting to do this.
4.  Print the original matrix, the mean vector, and the normalized (mean-subtracted) matrix.
5.  Verify that the column means of the new matrix are very close to zero.

---

### Exercise 2: Creating an Outer Product

**Task:**

The outer product of two vectors `a` and `b` results in a matrix where the element at `(i, j)` is `a[i] * b[j]`. You can create this using broadcasting.

1.  Create two 1D NumPy arrays, `a = np.arange(1, 5)` and `b = np.arange(1, 6)`.
2.  The goal is to multiply them to get a 4x5 matrix. If you just do `a * b`, it will fail due to incompatible shapes.
3.  Use `np.newaxis` to reshape `a` into a column vector of shape `(4, 1)`.
4.  Now, multiply the reshaped `a` (shape `(4, 1)`) with the original `b` (shape `(5,)`).
5.  Analyze why this works according to the rules of broadcasting.
    -   `a_reshaped.shape` -> `(4, 1)`
    -   `b.shape` -> `(5,)` -> padded to `(1, 5)`
    -   Dimension 1: 4 vs 1 -> 1 is stretched to 4.
    -   Dimension 2: 1 vs 5 -> 1 is stretched to 5.
    -   Resulting shape: `(4, 5)`
6.  Print the final outer product matrix. It should be a multiplication table.

---

### Exercise 3: Broadcasting with 3D Arrays

**Task:**

Let's apply broadcasting to a 3D array, which can be thought of as a collection of matrices (e.g., channels in an image).

1.  Create a 3D array of shape `(3, 4, 5)` filled with ones. This represents 3 matrices, each 4x5.
2.  Create a 1D array of shape `(5,)` with values `[10, 20, 30, 40, 50]`.
3.  Add the 1D array to the 3D array.
4.  Analyze the broadcasting rules:
    -   `arr_3d.shape` -> `(3, 4, 5)`
    -   `arr_1d.shape` -> `(5,)` -> padded to `(1, 1, 5)`
    -   Dimension 1: 3 vs 1 -> 1 is stretched.
    -   Dimension 2: 4 vs 1 -> 1 is stretched.
    -   Dimension 3: 5 vs 5 -> Match.
5.  Now, create a 2D array of shape `(4, 5)` with random values.
6.  Add this 2D array to the original 3D array. Analyze how it broadcasts across the first dimension (the "layers").
