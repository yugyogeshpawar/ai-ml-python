# Interview Questions: Array Broadcasting

---

### Question 1: What is NumPy broadcasting and what problem does it solve?

**Answer:**

**Broadcasting** is a set of rules that NumPy uses to allow operations on arrays of different, but compatible, shapes. It solves the problem of needing to perform arithmetic between arrays of mismatched sizes without having to manually reshape or create copies of the smaller array to match the larger one.

For example, instead of manually creating a new array `[5, 5, 5]` to add the scalar `5` to the array `[1, 2, 3]`, broadcasting allows you to simply write `[1, 2, 3] + 5`. NumPy handles the "stretching" of the smaller array (the scalar) virtually, without using extra memory, to perform the element-wise operation. This makes code cleaner, more intuitive, and highly memory-efficient.

---

### Question 2: You have a 2D array `A` with shape `(10, 5)` and a 1D array `b` with shape `(5,)`. Explain, according to the rules of broadcasting, why the operation `A + b` is valid.

**Answer:**

The operation `A + b` is valid because the shapes are compatible according to NumPy's broadcasting rules. Here is the step-by-step analysis:

1.  **Initial Shapes:**
    -   `A.shape` -> `(10, 5)`
    -   `b.shape` -> `(5,)`

2.  **Rule 1: Pad Dimensions.**
    NumPy compares the dimensions of the arrays starting from the trailing (rightmost) dimensions. Since array `b` has fewer dimensions than `A`, its shape is padded on the left with a `1`.
    -   `A.shape` -> `(10, 5)`
    -   `b.shape` -> `(1, 5)`

3.  **Rule 2: Compare Dimensions and Stretch.**
    Now, the dimensions are compared one by one.
    -   **Trailing dimension (axis 1):** `A` has size 5, `b` has size 5. They match. This is valid.
    -   **Leading dimension (axis 0):** `A` has size 10, `b` has size 1. Since one of the dimensions is 1, NumPy "stretches" or broadcasts `b` along this dimension to match the size of `A`. So, `b` is treated as if it were repeated 10 times along the first axis.

4.  **Conclusion:**
    Since all dimensions are compatible (they either match or one of them is 1), the operation is valid. The 1D array `b` is effectively added to every row of the 2D array `A`.

---

### Question 3: You have a 2D array `M` with shape `(4, 3)`. You want to subtract a 1D array `v` of shape `(4,)` from it. The operation `M - v` fails. Why does it fail, and how would you correct it?

**Answer:**

The operation `M - v` fails because the shapes `(4, 3)` and `(4,)` are not compatible for broadcasting. Here's the analysis:

1.  **Initial Shapes:**
    -   `M.shape` -> `(4, 3)`
    -   `v.shape` -> `(4,)`

2.  **Rule 1: Pad Dimensions.**
    The shape of `v` is padded on the left to match the number of dimensions of `M`.
    -   `M.shape` -> `(4, 3)`
    -   `v.shape` -> `(1, 4)`

3.  **Rule 2 & 3: Compare Dimensions.**
    -   **Trailing dimension (axis 1):** `M` has size 3, `v` has size 4. They do not match, and *neither is 1*.
    -   This disagreement immediately makes the shapes incompatible, and NumPy raises a `ValueError`.

**How to correct it:**

The intent is likely to subtract the vector `v` from each *column* of `M`. To do this, `v` needs to be aligned with the rows of `M`. This means `v` must be reshaped into a **column vector** of shape `(4, 1)`.

You can achieve this using `np.newaxis`:
`v_reshaped = v[:, np.newaxis]`

Now, let's re-evaluate the operation `M - v_reshaped`:
1.  **New Shapes:**
    -   `M.shape` -> `(4, 3)`
    -   `v_reshaped.shape` -> `(4, 1)`

2.  **Rule 2: Compare Dimensions.**
    -   **Trailing dimension (axis 1):** `M` is 3, `v_reshaped` is 1. The dimension of size 1 is stretched to match 3.
    -   **Leading dimension (axis 0):** `M` is 4, `v_reshaped` is 4. They match.

The shapes are now compatible. The column vector `v_reshaped` will be broadcast across the 3 columns of `M`, performing the subtraction as intended.
