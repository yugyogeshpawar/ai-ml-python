# Part 3, Topic 4: Integrating with Other Libraries

NumPy is the foundation of the scientific Python ecosystem. Its true power is fully realized when it's used as the data backbone for other powerful libraries like Pandas, Matplotlib, and Scikit-learn. The `ndarray` is the "lingua franca" that allows these different packages to communicate and work together seamlessly.

---

## 1. NumPy and Pandas

**Pandas** is the primary library for data manipulation and analysis in Python. Its core data structures, the `Series` (1D) and `DataFrame` (2D), are built directly on top of NumPy arrays.

### From NumPy to Pandas
You can easily create a Pandas DataFrame from a NumPy array.

```python
import numpy as np
import pandas as pd

# Create a NumPy array
data = np.random.standard_normal((5, 3))

# Create a DataFrame, adding column and index labels
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])

print(df)
```

### From Pandas to NumPy
You can extract the underlying NumPy array from a DataFrame or Series using the `.to_numpy()` method (or the `.values` attribute in older versions).

```python
# Get the NumPy array back from the DataFrame
numpy_array = df.to_numpy()

print(type(numpy_array)) # -> <class 'numpy.ndarray'>
```
This interoperability is crucial. You can use Pandas for its powerful data cleaning, alignment, and labeling features, and then easily extract the raw numerical data as a NumPy array to feed into a machine learning model.

---

## 2. NumPy and Matplotlib

**Matplotlib** is the most widely used library for plotting and data visualization in Python. It is designed to work directly with NumPy arrays. Most Matplotlib plotting functions expect NumPy arrays as their primary input.

```python
import matplotlib.pyplot as plt

# Create data using NumPy
x = np.linspace(0, 2 * np.pi, 100) # 100 points from 0 to 2*pi
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the data using Matplotlib
plt.figure(figsize=(8, 5))
plt.plot(x, y_sin, label='sin(x)')
plt.plot(x, y_cos, label='cos(x)')
plt.title('Sine and Cosine Waves')
plt.xlabel('x (radians)')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```
In this example, NumPy generates the structured numerical data, and Matplotlib visualizes it.

---

## 3. NumPy and Scikit-learn

**Scikit-learn** is the go-to library for machine learning in Python. Its entire API is designed around NumPy arrays. When you train a model, you pass NumPy arrays representing your features (`X`) and your target variable (`y`).

```python
from sklearn.linear_model import LinearRegression

# Create some sample data with NumPy
# X should be a 2D array (n_samples, n_features)
X = np.array([[1], [2], [3], [4]])
# y is the target variable
y = np.array([2, 3.9, 6.1, 8])

# Create and train a machine learning model
model = LinearRegression()
model.fit(X, y) # The .fit() method expects NumPy arrays

# Make a prediction
new_data = np.array([[5]])
prediction = model.predict(new_data)

print(f"Prediction for input {new_data[0][0]}: {prediction[0]:.2f}")
```
The seamless flow of data from NumPy to Scikit-learn is what makes the Python data science stack so powerful and cohesive. You can perform complex numerical transformations in NumPy and then directly use the results to train sophisticated machine learning models.
