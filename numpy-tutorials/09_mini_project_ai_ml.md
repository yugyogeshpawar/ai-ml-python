# 9. Putting it all together: A Mini-Project for AI/ML

Congratulations on making it to the final tutorial! We've covered a lot of ground, from creating arrays and understanding their attributes to performing element-wise operations, broadcasting, advanced indexing, and linear algebra.

In this tutorial, we'll apply these concepts to a practical mini-project relevant to AI/ML: **Data Normalization and Distance Calculation**. These are common preprocessing steps and analytical tasks in machine learning.

## Project Goal

You will:
1.  Generate a synthetic dataset.
2.  Normalize the dataset using Min-Max Scaling.
3.  Calculate the Euclidean distance between data points.

## 9.1 Step 1: Generate a Synthetic Dataset

Let's create a simple 2D dataset representing, for example, two features (e.g., 'Age' and 'Income') for 10 different samples.

```python
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 10 samples, each with 2 features
# Feature 1 (e.g., Age): values between 20 and 60
# Feature 2 (e.g., Income): values between 30000 and 100000
data = np.column_stack((np.random.randint(20, 61, 10),
                        np.random.randint(30000, 100001, 10)))

print("Original Dataset (10 samples, 2 features):\n", data)
print("Shape:", data.shape)
```

## 9.2 Step 2: Normalize the Dataset (Min-Max Scaling)

Normalization is a crucial preprocessing step in machine learning. Min-Max Scaling transforms features to a common scale, typically between 0 and 1. This prevents features with larger values from dominating the learning process.

The formula for Min-Max Scaling for a value `x` in a feature column is:
`x_normalized = (x - min_value) / (max_value - min_value)`

You need to calculate the minimum and maximum values for *each feature (column)* independently.

```python
import numpy as np

# (Assume 'data' from Step 1 is available)
np.random.seed(42)
data = np.column_stack((np.random.randint(20, 61, 10),
                        np.random.randint(30000, 100001, 10)))

print("Original Dataset:\n", data)

# Calculate min and max for each feature (column)
# axis=0 ensures we get min/max for each column
min_values = np.min(data, axis=0)
max_values = np.max(data, axis=0)

print("\nMinimum values per feature:", min_values)
print("Maximum values per feature:", max_values)

# Perform Min-Max Scaling using broadcasting
# (data - min_values) will broadcast min_values across rows
# (max_values - min_values) will also broadcast
normalized_data = (data - min_values) / (max_values - min_values)

print("\nNormalized Dataset (Min-Max Scaling):\n", normalized_data)
print("Min of normalized data:", np.min(normalized_data, axis=0)) # Should be close to 0
print("Max of normalized data:", np.max(normalized_data, axis=0)) # Should be close to 1
```

## 9.3 Step 3: Calculate Euclidean Distance

The Euclidean distance is a common metric used to measure the similarity or dissimilarity between two data points (vectors). It's the straight-line distance between two points in Euclidean space.

For two points `p = (p1, p2, ..., pn)` and `q = (q1, q2, ..., qn)`, the Euclidean distance is:
`d(p, q) = sqrt((p1-q1)^2 + (p2-q2)^2 + ... + (pn-qn)^2)`

In NumPy, this can be efficiently calculated using element-wise operations and `np.sum()` or `np.linalg.norm()`.

Let's calculate the distance between the first and second sample in our *normalized* dataset.

```python
import numpy as np

# (Assume 'normalized_data' from Step 2 is available)
np.random.seed(42)
data = np.column_stack((np.random.randint(20, 61, 10),
                        np.random.randint(30000, 100001, 10)))
min_values = np.min(data, axis=0)
max_values = np.max(data, axis=0)
normalized_data = (data - min_values) / (max_values - min_values)

# Select the first two samples
sample1 = normalized_data[0]
sample2 = normalized_data[1]

print("Sample 1 (normalized):", sample1)
print("Sample 2 (normalized):", sample2)

# Method 1: Manual calculation using element-wise operations
difference = sample1 - sample2
squared_difference = difference ** 2
sum_squared_difference = np.sum(squared_difference)
euclidean_distance_manual = np.sqrt(sum_squared_difference)
print("\nEuclidean Distance (Manual):", euclidean_distance_manual)

# Method 2: Using np.linalg.norm() (preferred for simplicity and efficiency)
euclidean_distance_norm = np.linalg.norm(sample1 - sample2)
print("Euclidean Distance (np.linalg.norm):", euclidean_distance_norm)
```

## Assignment: Mini-Project Extension

1.  **Generate a larger dataset:** Create a dataset with 50 samples and 3 features. The features can be random integers within different ranges (e.g., Feature 1: 0-10, Feature 2: 100-200, Feature 3: 1000-5000).
2.  **Apply Min-Max Scaling:** Normalize this new dataset using the Min-Max Scaling technique you learned. Print the first 5 rows of the normalized dataset and verify its min/max values.
3.  **Calculate distances for a specific point:**
    *   Choose the 5th sample (index 4) from your *normalized* dataset as a reference point.
    *   Calculate the Euclidean distance between this reference point and *all other samples* in the normalized dataset.
    *   Store these distances in a 1D NumPy array.
    *   Print the array of distances.
    *   (Hint: You can use broadcasting for the subtraction `(all_data - reference_point)` and then `np.linalg.norm` with `axis=1` to get distances for each row.)
4.  **Find the closest sample:** Determine which sample (by its original index) in the normalized dataset is closest to your reference point (excluding the reference point itself). Print its index and the distance.

---

This concludes our NumPy tutorial series! You now have a solid foundation in NumPy, which is an indispensable tool for anyone working in AI and Machine Learning. Keep practicing, and explore more advanced topics as you continue your journey!
