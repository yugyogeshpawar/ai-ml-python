# Interview Questions: Statistical Methods

---

### Question 1: In NumPy, what is the purpose of the `axis` argument in statistical functions like `np.sum()` or `np.mean()`? Explain what `axis=0` and `axis=1` mean for a 2D array.

**Answer:**

The `axis` argument specifies the dimension of the array that will be "collapsed" when the statistical function is computed. It allows you to perform aggregations along specific dimensions (e.g., down rows or across columns) rather than on the entire array.

For a 2D array (a matrix):
-   **`axis=0`**: This collapses the **rows**. The operation is performed "down the columns." For example, `arr.sum(axis=0)` will compute the sum of each column, resulting in a 1D array whose length is the number of columns.
-   **`axis=1`**: This collapses the **columns**. The operation is performed "across the rows." For example, `arr.sum(axis=1)` will compute the sum of each row, resulting in a 1D array whose length is the number of rows.

A good way to remember this is that the `axis` specified is the dimension that is *removed* from the output's shape. If you start with a `(3, 4)` array and sum along `axis=0`, you remove the dimension of size 3, leaving a `(4,)` shaped array.

---

### Question 2: What is the difference between `arr.argmax()` and `arr.max()`?

**Answer:**

The difference lies in what they return:

-   **`arr.max()`**: This function returns the **maximum value** contained within the array (or along a specified axis).
-   **`arr.argmin()`**: This function returns the **index** of the first occurrence of the maximum value within the array (or along a specified axis).

**Example:**
For `arr = np.array([10, 50, 20, 50])`:
-   `arr.max()` would return `50`.
-   `arr.argmax()` would return `1`, which is the index of the *first* time the maximum value (50) appears.

This distinction is crucial. You use `.max()` when you need to know what the highest value is, and you use `.argmax()` when you need to know *where* that highest value is located.

---

### Question 3: How can you use NumPy's statistical functions on a boolean array? What do `arr.sum()`, `arr.any()`, and `arr.all()` do on a boolean array `arr`?

**Answer:**

When statistical functions are applied to a boolean array, NumPy treats `True` as `1` and `False` as `0`. This provides a very powerful and efficient way to work with conditions.

-   **`arr.sum()`**: This will **count the number of `True` values** in the array. Since `True` is 1 and `False` is 0, the sum is equivalent to the count of `True` elements. This is a common idiom for counting how many elements in an array satisfy a certain condition.
    -   Example: `(data > 10).sum()` counts how many items in `data` are greater than 10.

-   **`arr.any()`**: This will return `True` if **at least one** element in the array is `True`. It is logically equivalent to asking "is there any `True` in the array?". It short-circuits and returns `True` on the first `True` value it finds.

-   **`arr.all()`**: This will return `True` only if **every single element** in the array is `True`. It is logically equivalent to asking "are all elements `True`?". It short-circuits and returns `False` on the first `False` value it finds.
