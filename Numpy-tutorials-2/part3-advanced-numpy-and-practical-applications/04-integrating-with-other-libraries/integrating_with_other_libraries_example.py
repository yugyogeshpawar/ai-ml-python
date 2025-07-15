# integrating_with_other_libraries_example.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# This script demonstrates a mini-project that uses NumPy, Pandas,
# Matplotlib, and Scikit-learn together.

# --- 1. Data Creation (NumPy) ---
# Create a synthetic dataset. We'll simulate a dataset for a classification
# problem: determining if a tumor is malignant based on two features.
print("--- 1. Creating data with NumPy ---")
rng = np.random.default_rng(seed=42)

# Class 0 (Benign)
class_0_features = rng.normal(loc=2, scale=1, size=(50, 2))

# Class 1 (Malignant)
class_1_features = rng.normal(loc=5, scale=1, size=(50, 2))

# Combine features and create labels
X = np.vstack((class_0_features, class_1_features))
y = np.array([0] * 50 + [1] * 50) # 50 labels for class 0, 50 for class 1

print("Shape of feature matrix X:", X.shape)
print("Shape of target vector y:", y.shape)
print("-" * 30)


# --- 2. Data Exploration (Pandas and Matplotlib) ---
print("\n--- 2. Exploring data with Pandas and Matplotlib ---")
# Convert to a Pandas DataFrame for easier manipulation and viewing
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Target'] = y
print("First 5 rows of the DataFrame:\n", df.head())

# Use Matplotlib to visualize the data
plt.figure(figsize=(8, 6))
# Scatter plot of benign tumors (Target == 0)
plt.scatter(df[df['Target'] == 0]['Feature_1'],
            df[df['Target'] == 0]['Feature_2'],
            c='b', label='Benign (Class 0)')
# Scatter plot of malignant tumors (Target == 1)
plt.scatter(df[df['Target'] == 1]['Feature_1'],
            df[df['Target'] == 1]['Feature_2'],
            c='r', label='Malignant (Class 1)')

plt.title('Synthetic Tumor Data Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
print("\nDisplaying data plot... (Close the plot window to continue)")
plt.show()
print("-" * 30)


# --- 3. Model Training (Scikit-learn) ---
print("\n--- 3. Training a model with Scikit-learn ---")
# The data is already in NumPy format (X and y), ready for Scikit-learn.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Initialize and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train) # Scikit-learn's .fit() uses NumPy arrays
print("\nModel training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

print("\nThis example shows the seamless workflow:")
print("NumPy -> Pandas -> Matplotlib -> Scikit-learn")
