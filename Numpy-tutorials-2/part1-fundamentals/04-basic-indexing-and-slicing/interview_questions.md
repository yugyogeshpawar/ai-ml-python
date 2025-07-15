# Interview Questions: Basic Indexing and Slicing

---

### Question 1: One of the most important concepts in NumPy is the difference between a "view" and a "copy." Explain this concept in the context of array slicing. Why is this design choice important for NumPy?

**Answer:**

In NumPy, when you take a slice of an array, you are not creating a new array with its own data in memory. Instead, you are creating a **view**.

-   A **view** is a new `ndarray` object that looks at the *same underlying data* as the original array. It's like a window or a pointer to a subset of the original data. If you modify the data through the view (e.g., `my_slice[0] = 100`), the changes will be reflected in the original array because they share the same data buffer.

-   A **copy**, on the other hand, is a completely new array with its own data duplicated in a new memory location. Modifying a copy has no effect on the original array. You must explicitly request a copy using the `.copy()` method (e.g., `my_copy = my_array[2:5].copy()`).

**Why this design choice is important:**
This design is centered around **performance and memory efficiency**. Scientific computing often involves very large datasets. If every slicing operation created a full copy of the data, it would be incredibly slow and consume vast amounts of memory, making it impractical to work with big data. By using views, NumPy avoids unnecessary data duplication, allowing for fast and memory-efficient data access and manipulation. The trade-off is that the programmer must be aware of this behavior to avoid unintentionally modifying the original data.

---

### Question 2: You have a 2D NumPy array called `data`. What is the difference in the output of `data[1]` and `data[:, 1]`?

**Answer:**

Although they might seem similar, these two expressions select completely different parts of the array:

-   **`data[1]`** (or `data[1, :]`): This selects the **entire row** at index 1. The result will be a **1D array** containing all the elements from the second row of the `data` matrix.

-   **`data[:, 1]`**: This selects the **entire column** at index 1. The colon `:` in the first position is a slice that means "select all rows." The `1` in the second position specifies the column index. The result will be a **1D array** containing the second element from every row.

**Example:**
Given `data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`:
-   `data[1]` would return `array([4, 5, 6])`.
-   `data[:, 1]` would return `array([2, 5, 8])`.

---

### Question 3: How would you select a 2x2 sub-matrix from the center of a 5x5 NumPy array `arr`?

**Answer:**

To select a 2x2 sub-matrix from the center of a 5x5 array, you need to identify the correct row and column indices.

A 5x5 array has indices from 0 to 4 for both rows and columns. The center of the array would involve rows and columns with indices 1, 2, and 3. A 2x2 sub-matrix would need two consecutive rows and two consecutive columns. The most central 2x2 block would be formed by rows at index 1 and 2, and columns at index 1 and 2.

The slicing expression would be:
`sub_matrix = arr[1:3, 1:3]`

**Explanation:**
-   `1:3` for the first axis (rows): This selects the rows at index 1 and 2 (the slice `stop` index, 3, is exclusive).
-   `1:3` for the second axis (columns): This selects the columns at index 1 and 2.

This single line of code will effectively extract the desired 2x2 central region of the matrix.
