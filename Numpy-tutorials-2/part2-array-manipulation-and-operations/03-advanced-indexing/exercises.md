# Exercises: Advanced Indexing

These exercises will challenge you to use boolean and fancy indexing to solve common data manipulation problems.

---

### Exercise 1: Filtering Data Points

**Task:**

You are given a 2D array of `(x, y)` coordinates. Your task is to select only the points that lie within a certain circular radius from the origin.

1.  Create a 100x2 NumPy array of random floating-point numbers between -10 and 10. This represents 100 `(x, y)` points.
2.  Calculate the distance of each point from the origin `(0, 0)`. The distance formula is `sqrt(x^2 + y^2)`. You can do this efficiently for all points at once using vectorized operations.
    -   Square the entire array.
    -   Sum the squares along the rows (axis=1).
    -   Take the square root of the result.
3.  Create a boolean mask to identify which points have a distance less than 5.
4.  Use this mask to select only the points that are "close" to the origin.
5.  Print the shape of the original points array and the shape of the filtered points array to see how many were selected.

---

### Exercise 2: Selecting and Reordering Matrix Rows and Columns

**Task:**

This exercise combines fancy indexing for both rows and columns.

1.  Create an 8x8 NumPy array with integers from 0 to 63.
2.  You want to "swap" the first and last columns. Create a new array where the first column of the original array is now the last, the last is now the first, and all other columns are in their original place.
    -   Hint: You can do this by creating a list of column indices in the desired order (e.g., `[7, 1, 2, 3, 4, 5, 6, 0]`) and using it for fancy indexing on the columns.
3.  From the *original* 8x8 array, you want to extract a sub-array containing the intersection of rows `[1, 3, 5]` and columns `[0, 2, 4, 6]`. Use fancy indexing on both axes to do this in a single step.
4.  Print the results of both steps.

---

### Exercise 3: Conditional Replacement

**Task:**

Use boolean indexing to find and replace values in an array based on a condition.

1.  Create a 1D NumPy array of integers from 0 to 20.
2.  You want to implement the classic "FizzBuzz" problem using NumPy.
    -   Create a new array of the same shape and `dtype=object` to store the results.
    -   Replace all numbers divisible by 3 with the string "Fizz".
    -   Replace all numbers divisible by 5 with the string "Buzz".
    -   Replace all numbers divisible by both 3 and 5 (i.e., by 15) with the string "FizzBuzz".
3.  To do this:
    -   Start by converting the integer array to an array of strings (`.astype(str)`).
    -   Use boolean indexing with the modulo operator (`%`) to find the correct positions for each replacement.
    -   **Important:** Perform the "FizzBuzz" replacement *first*, then "Fizz", then "Buzz". Why does the order matter?
4.  Print the final "FizzBuzz" array.
