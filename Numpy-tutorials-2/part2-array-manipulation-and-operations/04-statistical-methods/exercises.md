# Exercises: Statistical Methods

These exercises will help you practice using NumPy's aggregate functions to analyze data.

---

### Exercise 1: Data Normalization (Z-score)

**Task:**

A common way to normalize data is to calculate the "z-score" for each data point. The formula is `z = (x - μ) / σ`, where `μ` is the mean of the data and `σ` is the standard deviation.

1.  Create a 10x5 NumPy array of random numbers. This represents a dataset with 10 samples and 5 features.
2.  For each **column** (feature), calculate its mean and standard deviation. You should have two 1D arrays as a result, each of shape `(5,)`.
3.  Using broadcasting, apply the z-score formula to the entire data matrix.
4.  The resulting "normalized" matrix should have a mean very close to 0 and a standard deviation very close to 1 for each column. Verify this by calculating the mean and standard deviation of the columns of your new matrix.

---

### Exercise 2: Finding the Bestseller

**Task:**

You are given a 2D array representing the monthly sales of 5 different products over 12 months.

1.  Create a 5x12 NumPy array with random integers between 50 and 500.
2.  Find the total number of sales for the entire year.
3.  Find the month with the highest total sales across all products. (Hint: first find the sum of sales for each month, then find the index of the maximum value).
4.  Find the product with the highest total annual sales. (Hint: find the sum of sales for each product, then find the index of the maximum value).
5.  For each product, find its best and worst sales month. You should have two 1D arrays of shape `(5,)` containing the indices of the best and worst months for each product.

---

### Exercise 3: Counting Above Average

**Task:**

This exercise combines boolean indexing with statistical methods.

1.  Create a 1D NumPy array with 20 random integer values between 1 and 100.
2.  Calculate the overall mean of the array.
3.  Use a boolean condition to create a mask that is `True` for all elements greater than the mean.
4.  Use the `.sum()` method on this boolean mask to count how many elements are above average.
5.  Now, use the mask to create a new array containing *only* the above-average values.
6.  Calculate the mean of this new, smaller array. Is it higher than the original mean? It should be!
