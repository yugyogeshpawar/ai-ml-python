# statistical_methods_example.py

import numpy as np

# --- 1. Basic Aggregations on 1D Arrays ---
print("--- Basic Aggregations ---")
arr_1d = np.arange(1, 11) # Array from 1 to 10
print("1D Array:", arr_1d)

print(f"Sum (arr.sum()): {arr_1d.sum()}")
print(f"Mean (arr.mean()): {arr_1d.mean()}")
print(f"Standard Deviation (arr.std()): {arr_1d.std():.2f}")
print(f"Minimum (arr.min()): {arr_1d.min()}")
print(f"Maximum (arr.max()): {arr_1d.max()}")
print(f"Index of Minimum (arr.argmin()): {arr_1d.argmin()}")
print(f"Index of Maximum (arr.argmax()): {arr_1d.argmax()}")
print("-" * 30)


# --- 2. Aggregations Along Axes in 2D Arrays ---
print("\n--- Aggregations Along Axes ---")
# Create a 3x4 array representing, e.g., sales data for 3 products over 4 quarters
sales_data = np.array([
    [250, 280, 320, 310], # Product A
    [150, 160, 140, 180], # Product B
    [400, 410, 450, 480]  # Product C
])
print("Sales Data (3 products, 4 quarters):\n", sales_data)

# Calculate total sales for each product (sum across the columns, axis=1)
total_sales_per_product = sales_data.sum(axis=1)
print(f"\nTotal sales per product (sum along axis=1): {total_sales_per_product}")

# Calculate average sales per quarter (mean down the rows, axis=0)
avg_sales_per_quarter = sales_data.mean(axis=0)
print(f"Average sales per quarter (mean along axis=0): {avg_sales_per_quarter}")

# Find the quarter with the highest sales for each product (argmax along axis=1)
best_quarter_per_product = sales_data.argmax(axis=1)
print(f"Best quarter index for each product (argmax along axis=1): {best_quarter_per_product}")
print("-" * 30)


# --- 3. Cumulative Sums and Products ---
print("\n--- Cumulative Operations ---")
arr = np.array([1, 2, 3, 4, 5])
print("Original array:", arr)
print("Cumulative sum (cumsum):", arr.cumsum())
print("Cumulative product (cumprod):", arr.cumprod())

# In a 2D array
print("\nCumulative sum in 2D array (axis=1):\n", sales_data.cumsum(axis=1))
print("-" * 30)


# --- 4. Statistical Methods on Boolean Arrays ---
print("\n--- Stats on Boolean Arrays ---")
# Generate some random data
random_data = np.random.randn(10) # 10 random numbers from a standard normal distribution
print("Random data:\n", np.round(random_data, 2))

# How many values are positive?
positive_count = (random_data > 0).sum()
print(f"\nNumber of positive values: {positive_count}")

# Are there any values greater than 2?
any_large_values = (random_data > 2).any()
print(f"Are there any values > 2? {any_large_values}")

# Are all values less than 3?
all_small_values = (np.abs(random_data) < 3).all()
print(f"Are all absolute values < 3? {all_small_values}")
print("-" * 30)
