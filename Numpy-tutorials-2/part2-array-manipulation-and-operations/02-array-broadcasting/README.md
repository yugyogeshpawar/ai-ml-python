# Part 2, Topic 2: Array Broadcasting

Broadcasting is arguably one of the most powerful and magical features of NumPy. It describes the set of rules by which NumPy allows universal functions (ufuncs) to operate on arrays of different, but compatible, shapes. In essence, it lets you perform arithmetic operations between a smaller array and a larger array without explicitly creating copies of the smaller array to match the larger one's shape.

---

## 1. The Problem: Operations on Mismatched Shapes

Without broadcasting, operations between two arrays are only possible if their shapes are identical.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
# This works because shapes are both (3,)
result = a + b # -> [11 22 33]
```

But what if you want to add a single number (a scalar) to every element in an array?

```python
arr = np.array([0, 1, 2])
scalar = 5
# Intuitively, we want the result [5, 6, 7]
result = arr + scalar
```
This works seamlessly in NumPy because of broadcasting. NumPy "stretches" or "duplicates" the scalar `5` to match the shape of `arr`, so the operation becomes `[0, 1, 2] + [5, 5, 5]`.

---

## 2. The Rules of Broadcasting

Broadcasting doesn't actually duplicate the data in memory, which makes it very efficient. It follows a strict set of rules to determine if two arrays are compatible.

**Rule 1:** If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.

**Rule 2:** If the shape of two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.

**Rule 3:** If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

### Example 1: Adding a 1D array to a 2D array

Let's analyze `M + v` where `M` has shape `(3, 4)` and `v` has shape `(4,)`.

```
M.shape -> (3, 4)
v.shape -> (4,)
```

1.  **Rule 1:** `v` has fewer dimensions. Its shape is padded on the left with a 1.
    `M.shape -> (3, 4)`
    `v.shape -> (1, 4)`

2.  **Rule 2:** We compare the shapes dimension by dimension.
    -   Dimension 1: `M` is 3, `v` is 1. `v` can be "stretched" to match `M`.
    -   Dimension 2: `M` is 4, `v` is 4. The shapes match.

3.  **Result:** The shapes are compatible. `v` is broadcast across the 3 rows of `M`.

```python
M = np.ones((3, 4))
v = np.arange(4)
# M.shape is (3, 4), v.shape is (4,)
# After broadcasting, M + v is computed as if v had shape (3, 4)
result = M + v
# print(result) ->
# [[1. 2. 3. 4.]
#  [1. 2. 3. 4.]
#  [1. 2. 3. 4.]]
```

### Example 2: Incompatible Shapes

Let's try to add an array of shape `(3, 4)` and an array of shape `(3,)`.

```
a.shape -> (3, 4)
b.shape -> (3,)
```

1.  **Rule 1:** `b` has fewer dimensions. Its shape is padded on the left.
    `a.shape -> (3, 4)`
    `b.shape -> (1, 3)`

2.  **Rule 2 & 3:** We compare dimensions.
    -   Dimension 1: `a` is 3, `b` is 1. `b` can be stretched to match.
    -   Dimension 2: `a` is 4, `b` is 3. The sizes disagree, and *neither is 1*.

3.  **Result:** A `ValueError` is raised. The arrays are incompatible.

To make this work, `b` would need to be reshaped to `(3, 1)` so it could be broadcast across the columns of `a`.

Broadcasting is a powerful mental model to have. It allows you to write clean, concise, and efficient code without creating unnecessary temporary arrays, and it's fundamental to almost all non-trivial NumPy operations.
