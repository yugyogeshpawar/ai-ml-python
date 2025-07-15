# Interview Questions: Linear Algebra with NumPy

---

### Question 1: What is the modern, preferred way to perform matrix multiplication in NumPy, and why is it preferred over older methods like `np.dot()`?

**Answer:**

The modern, preferred way to perform matrix multiplication is using the infix operator **`@`**.

It is preferred over `np.dot()` for matrix multiplication primarily for **readability and clarity**. The `@` symbol is visually distinct and was specifically introduced to the language (in Python 3.5) to represent matrix multiplication, making the code's intent clearer to human readers, especially in complex formulas.

While `np.dot()` is a more general function that can perform dot products on 1D arrays (vectors) and matrix-vector multiplication, using `@` for matrix-matrix multiplication makes the code less ambiguous and more aligned with standard mathematical notation.

---

### Question 2: You need to solve a system of linear equations `Ax = b`. What are the two main ways to do this in NumPy, and which one is generally recommended?

**Answer:**

The two main ways to solve `Ax = b` are:

1.  **Using the Matrix Inverse:** Mathematically, the solution is `x = A_inv @ b`, where `A_inv` is the inverse of `A`. In NumPy, this would be coded as `x = np.linalg.inv(A) @ b`.

2.  **Using `np.linalg.solve()`:** This function is specifically designed to solve systems of linear equations. The code is `x = np.linalg.solve(A, b)`.

The **recommended method is `np.linalg.solve()`**.

**Reasoning:**
Calculating the matrix inverse explicitly with `np.linalg.inv()` is often less numerically stable and computationally more expensive than using a direct solver. `np.linalg.solve()` uses more efficient and stable algorithms (like LU decomposition) under the hood. It avoids potential precision issues that can arise from inverting a nearly-singular matrix and is generally faster.

---

### Question 3: What does it mean for a matrix to be "singular"? How can you check for this condition in NumPy, and what is the implication for linear algebra operations?

**Answer:**

A **singular matrix** is a square matrix that does not have an inverse. This means it cannot be "undone" by another matrix multiplication.

The most common way to check if a matrix is singular is by calculating its **determinant**. A matrix is singular if and only if its determinant is zero. In NumPy, you can check this with `np.linalg.det(A)`. Due to floating-point inaccuracies, it's best to check if the determinant is very close to zero, not just exactly zero (e.g., `np.isclose(np.linalg.det(A), 0)`).

**Implications for Linear Algebra:**
1.  **No Inverse:** A singular matrix cannot be inverted. Attempting to call `np.linalg.inv()` on a singular matrix will raise a `LinAlgError`.
2.  **No Unique Solution:** If the matrix `A` in the system `Ax = b` is singular, the system does not have a unique solution. It may have either no solutions or infinitely many solutions. Consequently, you cannot use `np.linalg.solve(A, b)` because it relies on `A` being invertible to find a unique solution.
