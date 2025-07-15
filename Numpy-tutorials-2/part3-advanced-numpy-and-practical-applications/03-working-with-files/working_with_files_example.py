# working_with_files_example.py

import numpy as np
import os

# --- 1. Saving and Loading a Single Array (.npy) ---
print("--- Single Array: .npy format ---")
# Create a sample array
my_array = np.arange(0, 5, 0.5).reshape(2, 5)
print("Original array:\n", my_array)

# Define the filename
npy_filename = 'single_array.npy'

# Save the array to the file
np.save(npy_filename, my_array)
print(f"\nArray saved to '{npy_filename}'")

# Load the array back from the file
loaded_array = np.load(npy_filename)
print(f"Array loaded from '{npy_filename}':\n", loaded_array)

# Check if the loaded array is identical to the original
print("Are original and loaded arrays identical?", np.array_equal(my_array, loaded_array))
print("-" * 30)


# --- 2. Saving and Loading Multiple Arrays (.npz) ---
print("\n--- Multiple Arrays: .npz format ---")
# Create a couple of arrays to save
mat_a = np.random.randn(3, 3)
vec_b = np.arange(10)
print("Matrix A:\n", mat_a)
print("Vector B:", vec_b)

# Define the filename
npz_filename = 'multi_array_archive.npz'

# Save the arrays using keyword arguments
np.savez(npz_filename, matrix=mat_a, vector=vec_b)
print(f"\nMultiple arrays saved to '{npz_filename}'")

# Load the archive
archive = np.load(npz_filename)

# Access the arrays using the keywords as keys
print("\nLoaded matrix from archive:\n", archive['matrix'])
print("Loaded vector from archive:", archive['vector'])
print("-" * 30)


# --- 3. Saving and Loading Text Files (.csv, .txt) ---
print("\n--- Text Files: .csv format ---")
# Create an integer array to save
data_to_save = np.arange(1, 21).reshape(4, 5)
print("Original data to save:\n", data_to_save)

# Define the filename
csv_filename = 'data.csv'

# Save to a CSV file
# We specify the format as integer ('%d') and the delimiter as a comma
np.savetxt(csv_filename, data_to_save, fmt='%d', delimiter=',')
print(f"\nData saved to '{csv_filename}'")

# Load the data back from the text file
loaded_data = np.loadtxt(csv_filename, delimiter=',')
print(f"Data loaded from '{csv_filename}':\n", loaded_data)
print("Data type of loaded data:", loaded_data.dtype) # Note: loadtxt defaults to float
print("-" * 30)


# --- Clean up the created files ---
print("\nCleaning up generated files...")
os.remove(npy_filename)
os.remove(npz_filename)
os.remove(csv_filename)
print("Files removed.")
