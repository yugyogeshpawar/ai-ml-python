# Exercises: Integrating with Other Libraries

These exercises will help you practice using NumPy as the bridge between different data science libraries.

---

### Exercise 1: From NumPy to Pandas and Back

**Task:**

You have a raw dataset in a NumPy array, but you need to use Pandas to add labels and perform some filtering before converting it back to a NumPy array for a machine learning model.

1.  Create a 10x4 NumPy array of random integers between 0 and 100. This is your raw data.
2.  Create a Pandas DataFrame from this array. Give the columns names: `['Metric1', 'Metric2', 'Metric3', 'Category']`.
3.  In Pandas, filter the DataFrame to select only the rows where the 'Category' column is greater than 50.
4.  From this filtered DataFrame, extract the first three columns (`'Metric1'`, `'Metric2'`, `'Metric3'`) and convert this subset back into a new NumPy array.
5.  Print the shape of the original NumPy array and the shape of the final, filtered NumPy array to see the result of your workflow.

---

### Exercise 2: Plotting a Histogram of NumPy Data

**Task:**

You want to visualize the distribution of a large set of random data points that you generate with NumPy.

1.  Use NumPy's random number generator to create a 1D array of 1,000 data points sampled from a normal distribution with a mean of 150 and a standard deviation of 25.
2.  Use Matplotlib's `plt.hist()` function to create a histogram of this data.
3.  Customize your plot:
    -   Add a title (e.g., "Distribution of Sample Data").
    -   Add labels for the x-axis ("Value") and y-axis ("Frequency").
    -   Change the number of bins to 30.
4.  Display the plot using `plt.show()`. This exercise demonstrates the direct pipeline from NumPy data generation to Matplotlib visualization.

---

### Exercise 3: A Minimal Scikit-learn Workflow

**Task:**

This exercise walks through the absolute basics of a machine learning workflow, emphasizing how NumPy arrays are used at each step.

1.  **Data Creation (NumPy):**
    -   Create a feature matrix `X` of shape `(100, 1)` containing numbers from 0 to 99. `np.arange(100).reshape(-1, 1)` is a good way to do this.
    -   Create a target vector `y` that is a linear function of `X` with some random noise. For example: `y = 2 * X.ravel() + 5 + rng.normal(0, 10, size=100)`. (`.ravel()` is used to make `y` a 1D array).
2.  **Model Training (Scikit-learn):**
    -   Import `LinearRegression` from `sklearn.linear_model`.
    -   Create an instance of the model.
    -   Call the `.fit()` method, passing your NumPy arrays `X` and `y`.
3.  **Prediction (Scikit-learn and NumPy):**
    -   Create a new NumPy array `X_new` with some values you want to predict, e.g., `np.array([[100], [101]])`.
    -   Use the trained model's `.predict()` method on `X_new` to get the predictions.
4.  Print the coefficients of the trained model (`model.coef_` and `model.intercept_`) and the predictions for `X_new`. The coefficients should be close to the `2` and `5` we used to generate the data.
