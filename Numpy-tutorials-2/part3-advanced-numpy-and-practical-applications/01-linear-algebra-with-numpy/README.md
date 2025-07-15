# Part 3, Topic 1: Linear Algebra with NumPy

Linear algebra is a foundational field of mathematics for data science and machine learning. NumPy's `linalg` module provides a comprehensive and efficient set of tools for performing linear algebra operations.

---

## 1. Matrix Multiplication

One of the most common linear algebra operations is matrix multiplication. NumPy provides multiple ways to do this, but the dedicated `@` operator (introduced in Python 3.5) is the preferred method.

For two matrices `A` and `B`, the product `A @ B` is valid only if the number of columns in `A` is equal to the number of rows in `B`.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])      # Shape (2, 2)
B = np.array([[5, 6], [7, 8]])      # Shape (2, 2)
# C has shape (3, 2), D has shape (2, 4)
C = np.arange(1, 7).reshape(3, 2)
D = np.arange(1, 9).reshape(2, 4)

# Matrix multiplication using the @ operator
print("A @ B:\n", A @ B)

# Multiplication of non-square matrices
# (3, 2) @ (2, 4) -> results in a (3, 4) matrix
print("\nC @ D:\n", C @ D)
```
The `@` operator is equivalent to the `np.dot` function for 2D arrays, but `@` is generally preferred for its readability in the context of matrix multiplication.

---

## 2. The `numpy.linalg` Module

For more advanced operations, we use functions from the `np.linalg` module.

### `np.linalg.inv()` - Matrix Inverse
The inverse of a square matrix `A` is another matrix `A_inv` such that `A @ A_inv` results in the identity matrix (a square matrix with ones on the diagonal and zeros elsewhere). A matrix must be square and non-singular (i.e., its determinant is non-zero) to have an inverse.

```python
A = np.array([[1., 2.], [3., 4.]])
A_inv = np.linalg.inv(A)

print("Inverse of A:\n", A_inv)

# Verification: A @ A_inv should be close to the identity matrix
# Due to floating point inaccuracies, it won't be perfect.
identity = np.eye(2)
print("\nA @ A_inv (should be identity matrix):\n", A @ A_inv)
# [[1.00000000e+00, 0.00000000e+00],
#  [8.88178420e-16, 1.00000000e+00]]
```

### `np.linalg.det()` - Determinant
The determinant is a scalar value that can be computed from the elements of a square matrix. It provides important information about the matrix (e.g., if it's invertible).

```python
det_A = np.linalg.det(A)
print("\nDeterminant of A:", det_A) # -> -2.0
```

### `np.linalg.eig()` - Eigenvalues and Eigenvectors
Eigenvalues and eigenvectors are a cornerstone of many machine learning algorithms, particularly in dimensionality reduction (like PCA). For a matrix `A`, an eigenvector `v` and its corresponding eigenvalue `λ` satisfy the equation `A @ v = λ * v`.

The `np.linalg.eig()` function returns a tuple containing a vector of eigenvalues and a matrix of the corresponding eigenvectors.

```python
# Create a symmetric matrix
S = np.array([[1, 2], [2, 1]])

eigenvalues, eigenvectors = np.linalg.eig(S)

print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verification for the first eigenvalue/eigenvector pair
v1 = eigenvectors[:, 0] # First column is the first eigenvector
lambda1 = eigenvalues[0]

# Check if S @ v1 is approximately equal to lambda1 * v1
print("\nS @ v1:", S @ v1)
print("lambda1 * v1:", lambda1 * v1)
```

These are just a few of the powerful functions available in `np.linalg`. Others include tools for solving systems of linear equations (`solve`), computing decompositions (`svd`, `qr`), and more.
