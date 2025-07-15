# linear_algebra_with_numpy_example.py

import numpy as np

# --- 1. Matrix Multiplication ---
print("--- Matrix Multiplication ---")
A = np.array([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)
B = np.array([[7, 8], [9, 10], [11, 12]]) # Shape (3, 2)

# The inner dimensions (3 and 3) match, so we can multiply them.
# The resulting shape will be the outer dimensions (2, 2).
C = A @ B
print(f"Matrix A (shape {A.shape}):\n{A}")
print(f"\nMatrix B (shape {B.shape}):\n{B}")
print(f"\nResult of A @ B (shape {C.shape}):\n{C}")
print("-" * 30)


# --- 2. Solving a System of Linear Equations ---
# Consider the system of equations:
# 2x + y = 8
#  x + 3y = 11
# This can be written in matrix form as A @ x = b
# where A = [[2, 1], [1, 3]], x = [x, y], b = [8, 11]

print("\n--- Solving Linear Equations ---")
A_eq = np.array([[2, 1], [1, 3]])
b_eq = np.array([8, 11])

# We can find the solution for x by calculating the inverse of A and multiplying by b
# x = A_inv @ b
try:
    A_inv = np.linalg.inv(A_eq)
    solution = A_inv @ b_eq
    print("Solution using inverse:", solution)
except np.linalg.LinAlgError:
    print("Matrix is singular and cannot be inverted.")

# A more stable and preferred way is to use np.linalg.solve()
# This avoids calculating the inverse directly, which can be numerically unstable.
solution_solve = np.linalg.solve(A_eq, b_eq)
print("Solution using np.linalg.solve():", solution_solve)
print("Verification: 2*2.6 + 2.8 = 8.0, 2.6 + 3*2.8 = 11.0")
print("-" * 30)


# --- 3. Determinant and Inverse ---
print("\n--- Determinant and Inverse ---")
M = np.array([[4, 7], [2, 6]])
print("Matrix M:\n", M)

# Determinant
det_M = np.linalg.det(M)
print(f"\nDeterminant of M: {det_M:.2f}") # 4*6 - 7*2 = 24 - 14 = 10

# Inverse
if det_M != 0:
    M_inv = np.linalg.inv(M)
    print("\nInverse of M:\n", M_inv)
    # Verification
    identity = M @ M_inv
    print("\nVerification (M @ M_inv):\n", np.round(identity))
else:
    print("\nMatrix M is singular, no inverse exists.")
print("-" * 30)


# --- 4. Eigenvalues and Eigenvectors ---
print("\n--- Eigenvalues and Eigenvectors ---")
# Create a symmetric matrix
S = np.array([[6, 2], [2, 3]])
print("Symmetric Matrix S:\n", S)

eigenvalues, eigenvectors = np.linalg.eig(S)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify the first eigenvector and eigenvalue
# A @ v = lambda * v
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0] # First column

print("\nVerification for first pair:")
print("S @ v1 =", S @ v1)
print("lambda1 * v1 =", lambda1 * v1)
# The results should be the same.
print("-" * 30)
