# Interview Questions: Advanced Indexing

---

### Question 1: What is the key difference in the output between basic slicing and advanced indexing (e.g., boolean or fancy indexing)?

**Answer:**

The key difference is that **basic slicing returns a *view* of the array, while advanced indexing returns a *copy* of the data.**

-   **View (Basic Slicing):** A slice like `arr[1:5, :]` creates a new array object that points to the *same underlying data* as the original array. Modifying the slice will modify the original array. This is a memory-efficient feature for accessing parts of an array without duplicating data.

-   **Copy (Advanced Indexing):** An index like `arr[arr > 5]` (boolean) or `arr[[0, 4, 2]]` (fancy) creates a brand new array with a completely new data buffer in memory. The data is copied from the original array into this new one. Modifying the new array will have no effect on the original.

This distinction is critical. If you need to modify a subset of an array without affecting the original, you must use advanced indexing or explicitly create a copy with `.copy()`. If you are just reading data, using views is more performant.

---

### Question 2: You have a 2D array `data`. Write a single line of code that selects all rows where the value in the first column is greater than 0.5.

**Answer:**

You can achieve this by using boolean indexing. The condition is based on the first column, and the resulting boolean array is used to index the rows of the original array.

The line of code would be:
`selected_data = data[data[:, 0] > 0.5]`

**Explanation:**
1.  `data[:, 0]` uses basic slicing to select the entire first column of the `data` array, resulting in a 1D array.
2.  `> 0.5` is a vectorized comparison that is applied to this 1D array, producing a 1D boolean array (a mask). The mask will have `True` for every row where the first column's value was greater than 0.5, and `False` otherwise.
3.  `data[...]` uses this boolean mask to perform boolean indexing on the `data` array. It selects only the rows where the mask has a `True` value.

---

### Question 3: What is "fancy indexing"? Provide an example of how you would use it to select the elements at coordinates `(0, 1)`, `(2, 3)`, and `(3, 0)` from a 4x4 matrix `M`.

**Answer:**

**Fancy indexing** is the term for using an array of integers to index another array. It allows you to select arbitrary elements, rows, or columns in any order you choose.

To select the elements at coordinates `(0, 1)`, `(2, 3)`, and `(3, 0)` from a matrix `M`, you provide two arrays of indices: one for the row positions and one for the column positions.

**Example:**
```python
import numpy as np

# Create a 4x4 matrix
M = np.arange(16).reshape(4, 4)

# Define the row and column indices for the desired elements
rows = np.array([0, 2, 3])
cols = np.array([1, 3, 0])

# Use fancy indexing to get the elements
selected_elements = M[rows, cols]

# print(selected_elements) will output: [ 1 11 12]
```
This works by pairing the indices from the `rows` and `cols` arrays. The first element selected is `M[rows[0], cols[0]]` which is `M[0, 1]`. The second is `M[rows[1], cols[1]]` which is `M[2, 3]`, and so on. The result is a 1D array containing the selected elements.
