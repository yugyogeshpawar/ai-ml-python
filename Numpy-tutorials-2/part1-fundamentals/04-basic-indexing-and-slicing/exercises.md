# Exercises: Basic Indexing and Slicing

These exercises will test your ability to select and manipulate specific parts of NumPy arrays.

---

### Exercise 1: Extracting Sub-regions

**Task:**

1.  Create a 6x6 NumPy array containing integers from 1 to 36. The easiest way is with `np.arange(1, 37).reshape((6, 6))`.
2.  From this array, use a single slicing operation to extract the following 3x3 sub-array:
    ```
    [[15 16 17]
     [21 22 23]
     [27 28 29]]
    ```
3.  Use another slicing operation to extract the four corner elements of the original 6x6 array. This is a bit trickier! You might need more than one line of code or a more advanced indexing method (which we'll cover later), but see if you can do it with basic slicing and `np.array()` construction.
4.  Extract the last column of the array as a 1D array.

---

### Exercise 2: The "View" vs. "Copy" Challenge

**Task:**

This exercise will solidify your understanding of the difference between a slice (view) and a copy.

1.  Create a 1D NumPy array with the numbers 0 through 9.
2.  Create a slice of this array containing the elements from index 3 to 7. Assign this slice to a variable called `my_view`.
3.  Create a copy of the same slice (indices 3 to 7) and assign it to a variable called `my_copy`.
4.  Multiply every element in `my_view` by 10.
5.  Multiply every element in `my_copy` by 100.
6.  Print the original array.
7.  Which modification affected the original array? Write a comment in your script explaining why.

---

### Exercise 3: Setting a Border to Zero

**Task:**

Given a 5x5 matrix of ones, write a script that uses slicing to set all the border elements to 0, leaving the inner 3x3 matrix as ones.

**Hint:**

-   Start with `arr = np.ones((5, 5))`.
-   You can solve this by setting each border (top row, bottom row, first column, last column) to zero one by one.
-   Can you think of a way to do this in fewer than four slicing assignments?

**Expected Output:**
```
[[0. 0. 0. 0. 0.]
 [0. 1. 1. 1. 0.]
 [0. 1. 1. 1. 0.]
 [0. 1. 1. 1. 0.]
 [0. 0. 0. 0. 0.]]
