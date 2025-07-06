# 8. Linear Algebra with NumPy

In the previous tutorial, we learned how to manipulate arrays by reshaping, stacking, and splitting them. Now, we'll delve into **Linear Algebra** operations using NumPy, which are absolutely fundamental to many AI/ML algorithms, including neural networks, principal component analysis (PCA), and regression.

NumPy's `linalg` module provides a comprehensive set of functions for linear algebra.

## 8.1 Matrix Multiplication

Matrix multiplication is one of the most common operations in linear algebra and machine learning.

### Using `np.dot()` or `@` operator

For 2D arrays (matrices), `np.dot()` performs matrix multiplication. For 1D arrays, it performs a dot product. The `@` operator (introduced in Python 3.5) is a more intuitive way to perform matrix multiplication.

**Rule for Matrix Multiplication:**
For two matrices A (m x n) and B (n x p), their product C (m x p) is defined if and only if the number of columns in A equals the number of rows in B.

```python
import numpy as np

# Define two matrices
matrix_a = np.array([[1, 2],
                     [3, 4]]) # Shape (2, 2)

matrix_b = np.array([[5, 6],
                     [7, 8]]) # Shape (2, 2)

print("Matrix A:\n", matrix_a)
print("Matrix B:\n", matrix_b)

# Matrix multiplication using np.dot()
result_dot = np.dot(matrix_a, matrix_b)
print("\nMatrix A . Matrix B (using np.dot):\n", result_dot)

# Matrix multiplication using the @ operator (preferred for clarity)
result_at = matrix_a @ matrix_b
print("Matrix A @ Matrix B (using @ operator):\n", result_at)

# Example with different shapes
matrix_c = np.array([[1, 2, 3],
                     [4, 5, 6]]) # Shape (2, 3)

matrix_d = np.array([[7, 8],
                     [9, 10],
                     [11, 12]]) # Shape (3, 2)

print("\nMatrix C:\n", matrix_c)
print("Matrix D:\n", matrix_d)

result_cd = matrix_c @ matrix_d
print("Matrix C @ Matrix D:\n", result_cd) # Resulting shape (2, 2)
```

**Important Note:** `*` performs element-wise multiplication, not matrix multiplication. Be careful not to confuse them!

## 8.2 Determinant of a Matrix

The determinant is a scalar value that can be computed from the elements of a square matrix. It provides important information about the matrix, such as whether it is invertible.

```python
import numpy as np

square_matrix = np.array([[1, 2],
                          [3, 4]])

print("Square Matrix:\n", square_matrix)

# Calculate the determinant
determinant = np.linalg.det(square_matrix)
print("Determinant:", determinant) # Output: -2.0 (1*4 - 2*3 = 4 - 6 = -2)

# Example with a 3x3 matrix
matrix_3x3 = np.array([[1, 2, 3],
                       [0, 1, 4],
                       [5, 6, 0]])
print("\n3x3 Matrix:\n", matrix_3x3)
print("Determinant of 3x3 Matrix:", np.linalg.det(matrix_3x3))
```

## 8.3 Inverse of a Matrix

The inverse of a square matrix `A`, denoted `A^-1`, is a matrix such that when multiplied by `A`, it yields the identity matrix (`I`). An inverse exists only if the determinant of the matrix is non-zero.

```python
import numpy as np

square_matrix = np.array([[1, 2],
                          [3, 4]])

print("Square Matrix:\n", square_matrix)

# Calculate the inverse
inverse_matrix = np.linalg.inv(square_matrix)
print("Inverse Matrix:\n", inverse_matrix)

# Verify: A @ A_inv should be close to identity matrix
identity_check = square_matrix @ inverse_matrix
print("\nMatrix @ Inverse (should be Identity):\n", identity_check)

# Example of a singular matrix (determinant is 0), which has no inverse
singular_matrix = np.array([[1, 2],
                            [2, 4]])
print("\nSingular Matrix:\n", singular_matrix)
print("Determinant of Singular Matrix:", np.linalg.det(singular_matrix)) # Output: ~0.0

try:
    np.linalg.inv(singular_matrix)
except np.linalg.LinAlgError as e:
    print(f"Error trying to invert singular matrix: {e}")
```

## 8.4 Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are special vectors and scalars associated with a linear transformation. They are crucial in many areas of AI/ML, such as PCA for dimensionality reduction, and in understanding the stability of systems.

For a square matrix `A`, an eigenvector `v` and its corresponding eigenvalue `λ` satisfy the equation:
`Av = λv`

```python
import numpy as np

matrix_e = np.array([[2, -1],
                     [-1, 2]])

print("Matrix for Eigenvalues/Eigenvectors:\n", matrix_e)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix_e)

print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors) # Each column is an eigenvector
```

**Explanation:**
*   `eigenvalues` is a 1D array containing the eigenvalues.
*   `eigenvectors` is a 2D array where each column is an eigenvector corresponding to the eigenvalue at the same index.

## 8.5 Solving Systems of Linear Equations

NumPy can efficiently solve systems of linear equations of the form `Ax = b`, where `A` is a matrix, `x` is the vector of unknowns, and `b` is a vector of constants.

Consider the system:
`x + 2y = 5`
`3x + 4y = 11`

This can be written in matrix form as:
`[[1, 2], [3, 4]] @ [[x], [y]] = [[5], [11]]`

```python
import numpy as np

# Define matrix A and vector b
A = np.array([[1, 2],
              [3, 4]])

b = np.array([5, 11])

print("Matrix A:\n", A)
print("Vector b:", b)

# Solve for x using np.linalg.solve()
x = np.linalg.solve(A, b)
print("\nSolution x (values for x and y):", x) # Output: [1. 2.] (meaning x=1, y=2)

# Verify the solution: A @ x should be equal to b
print("Verification (A @ x):", A @ x)
```

`np.linalg.solve()` is generally preferred over calculating the inverse and then multiplying (`np.linalg.inv(A) @ b`) because it is more numerically stable and often faster.

## Practical Applications in AI/ML

*   **Regression:** Solving for coefficients in linear regression models.
*   **Optimization:** Many optimization algorithms involve solving linear systems.
*   **Dimensionality Reduction:** PCA relies heavily on eigenvalues and eigenvectors.
*   **Neural Networks:** Matrix multiplication is the core operation in neural network layers.

## Assignment: Linear Algebra Practice

1.  Create two matrices: `M1` (3x2) and `M2` (2x4) with random integers.
2.  Perform matrix multiplication `M1 @ M2`. Print the result and its shape.
3.  Create a 3x3 identity matrix using `np.eye(3)`.
4.  Calculate the determinant of the identity matrix. What do you observe?
5.  Create a 2x2 matrix `P = np.array([[4, 2], [1, 3]])`.
6.  Calculate the inverse of `P`.
7.  Verify the inverse by multiplying `P` with its inverse.
8.  Solve the following system of linear equations:
    `2x + y = 7`
    `x - 3y = -7`

---

In the final tutorial, we will work on a mini-project to apply the NumPy concepts learned throughout this series to an AI/ML context.

**Next:** [9. Putting it all together: A Mini-Project for AI/ML](09_mini_project_ai_ml.md)
**Previous:** [7. Manipulating Arrays](07_manipulating_arrays.md)
