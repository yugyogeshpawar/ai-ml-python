# 6. Advanced Indexing

In the previous tutorial, we learned about Broadcasting, a powerful feature for performing operations on arrays of different shapes. Now, we'll explore **Advanced Indexing** techniques in NumPy, specifically **Boolean Indexing** and **Fancy Indexing**. These methods allow you to select non-contiguous or conditional subsets of your data, which is incredibly useful in AI/ML for data filtering and manipulation.

## 6.1 Boolean Indexing

Boolean indexing (also known as masking) allows you to select elements from an array based on a boolean condition. You provide a boolean array of the same shape as the array you are indexing, where `True` values correspond to the elements you want to select, and `False` values correspond to elements you want to ignore.

### Example 1: Selecting elements based on a condition

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70])

# Create a boolean array based on a condition
# This will be [False, False, True, True, True, True, True]
condition = arr > 25
print("Original Array:", arr)
print("Condition (arr > 25):", condition)

# Use the boolean array to select elements
selected_elements = arr[condition]
print("Elements > 25:", selected_elements) # Output: [30 40 50 60 70]

# You can combine the condition directly
print("Elements > 50:", arr[arr > 50]) # Output: [60 70]
```

### Example 2: Boolean indexing in 2D arrays

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("\nOriginal Matrix:\n", matrix)

# Select elements greater than 5
greater_than_5 = matrix[matrix > 5]
print("Elements > 5:", greater_than_5) # Output: [6 7 8 9] (returns a 1D array)

# Select elements that are even
even_elements = matrix[matrix % 2 == 0]
print("Even elements:", even_elements) # Output: [2 4 6 8]

# Combine multiple conditions (use & for AND, | for OR)
# Select elements greater than 3 AND less than 8
condition_combined = (matrix > 3) & (matrix < 8)
print("\nCondition (3 < element < 8):\n", condition_combined)
# Output:
# [[False False False]
#  [ True  True  True]
#  [ True False False]]

print("Elements (3 < element < 8):", matrix[condition_combined]) # Output: [4 5 6 7]
```

### Example 3: Modifying elements using Boolean Indexing

Boolean indexing can also be used to modify specific elements that meet a condition.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print("\nOriginal Array for modification:", arr)

# Set all elements greater than 30 to 0
arr[arr > 30] = 0
print("Array after setting elements > 30 to 0:", arr) # Output: [10 20 30  0  0]

# Set specific elements based on a condition
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("\nOriginal Matrix for modification:\n", matrix)

matrix[matrix % 2 != 0] = -1 # Set odd numbers to -1
print("Matrix after setting odd numbers to -1:\n", matrix)
# Output:
# [[-1  2 -1]
#  [ 4 -1  6]
#  [-1  8 -1]]
```

## 6.2 Fancy Indexing

Fancy indexing allows you to select specific rows, columns, or elements using arrays of integer indices. This is different from slicing, which only works for contiguous blocks.

### Example 1: Selecting non-contiguous elements (1D array)

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

# Select elements at specific indices
indices = np.array([0, 3, 5])
selected_elements = arr[indices]
print("\nOriginal Array:", arr)
print("Indices to select:", indices)
print("Elements at specified indices:", selected_elements) # Output: [10 40 60]

# You can also use a Python list of integers
print("Elements using Python list of indices:", arr[[1, 4]]) # Output: [20 50]
```

### Example 2: Selecting rows in a 2D array

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

print("\nOriginal Matrix:\n", matrix)

# Select specific rows
row_indices = np.array([0, 2]) # Select first and third row
selected_rows = matrix[row_indices]
print("Selected rows (0 and 2):\n", selected_rows)
# Output:
# [[1 2 3]
#  [7 8 9]]

# Select rows in a specific order, or with repetition
ordered_rows = matrix[[3, 1, 3]]
print("Selected rows in custom order:\n", ordered_rows)
# Output:
# [[10 11 12]
#  [ 4  5  6]
#  [10 11 12]]
```

### Example 3: Selecting columns in a 2D array

To select columns, you need to use slicing for the rows and fancy indexing for the columns.

```python
import numpy as np

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print("\nOriginal Matrix:\n", matrix)

# Select specific columns (e.g., column 0 and 2)
col_indices = np.array([0, 2])
selected_columns = matrix[:, col_indices]
print("Selected columns (0 and 2):\n", selected_columns)
# Output:
# [[ 1  3]
#  [ 5  7]
#  [ 9 11]]
```

### Example 4: Combining Fancy Indexing for elements

You can use two arrays of indices (one for rows, one for columns) to select specific elements at corresponding (row, column) pairs.

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("\nOriginal Matrix:\n", matrix)

# Select elements at (0,0), (1,2), (2,1)
row_coords = np.array([0, 1, 2])
col_coords = np.array([0, 2, 1])

selected_elements = matrix[row_coords, col_coords]
print("Elements at (0,0), (1,2), (2,1):", selected_elements) # Output: [1 6 8]
```

## Practical Applications in AI/ML

*   **Data Filtering:** Easily select data points that meet certain criteria (e.g., all samples where a feature is above a threshold).
*   **Feature Selection:** Extract specific columns (features) from a dataset.
*   **Sampling:** Randomly select a subset of data for training or validation.
*   **Data Cleaning:** Identify and modify outliers or missing values.

## Assignment: Advanced Indexing Practice

1.  Create a 1D array `data` with 20 random integers between 1 and 100.
2.  Using Boolean Indexing, select and print all numbers in `data` that are greater than 50.
3.  Using Boolean Indexing, change all numbers in `data` that are less than 20 to 0. Print the modified array.
4.  Create a 2D array `grades` of shape (5, 3) representing 5 students and 3 subjects, filled with random integers between 50 and 100.
5.  Using Fancy Indexing, select and print the grades of students at index 0, 2, and 4.
6.  Using Fancy Indexing, select and print the grades for subject 1 (index 0) and subject 3 (index 2) for all students.

---

In the next tutorial, we will learn about manipulating arrays, including reshaping, stacking, and splitting.
