# Exercises: Linear Algebra with NumPy

These exercises will help you practice common linear algebra operations using NumPy.

---

### Exercise 1: Solving a System of Linear Equations

**Task:**

You are an economist modeling supply and demand. You have the following equations:

-   Supply: `Qs = 5 + 2*P`
-   Demand: `Qd = 50 - 3*P`

In equilibrium, `Qs = Qd`. We can rewrite this as a system of linear equations to solve for the equilibrium price `P` and quantity `Q`:

1.  `Q - 2*P = 5`
2.  `Q + 3*P = 50`

Represent this system in the matrix form `A @ x = b`, where `x` is the vector `[Q, P]`.

1.  Define the matrix `A` and the vector `b`.
2.  Use `np.linalg.solve()` to find the solution vector `x`.
3.  Print the equilibrium quantity `Q` and price `P`.
4.  Verify your answer by plugging the values back into the original supply and demand equations.

---

### Exercise 2: Checking for Singularity

**Task:**

A matrix is "singular" if its determinant is zero. A singular matrix does not have an inverse, which means it cannot be used to uniquely solve a system of linear equations.

1.  Create two 2x2 matrices:
    -   `A = np.array([[1, 2], [3, 4]])`
    -   `B = np.array([[1, 2], [2, 4]])`
2.  Calculate the determinant of both `A` and `B` using `np.linalg.det()`.
3.  Write a simple script that checks if each matrix is singular.
4.  For the non-singular matrix, calculate its inverse using `np.linalg.inv()`.
5.  For the singular matrix, try to calculate its inverse and observe the `LinAlgError` that NumPy raises. Use a `try...except` block to catch this error and print a user-friendly message.

---

### Exercise 3: Eigen-decomposition and Reconstruction

**Task:**

A powerful property of symmetric matrices is that they can be reconstructed from their eigenvalues and eigenvectors. The formula for reconstruction is `A = V @ diag(λ) @ V_inv`, where `V` is the matrix of eigenvectors, `diag(λ)` is a diagonal matrix of eigenvalues, and `V_inv` is the inverse of `V`.

1.  Create a 3x3 symmetric matrix `S`. A simple way is `A = np.random.rand(3,3); S = A + A.T`.
2.  Calculate the eigenvalues (`evals`) and eigenvectors (`evecs`) of `S` using `np.linalg.eig()`.
3.  Reconstruct the original matrix `S` using the formula:
    -   Create a diagonal matrix from the eigenvalues using `np.diag(evals)`.
    -   Calculate the inverse of the eigenvector matrix, `np.linalg.inv(evecs)`.
    -   Multiply the three matrices together using the `@` operator.
4.  Use `np.allclose()` to compare your reconstructed matrix with the original matrix `S`. This function is better than `==` for comparing floating-point arrays as it allows for a small tolerance.
