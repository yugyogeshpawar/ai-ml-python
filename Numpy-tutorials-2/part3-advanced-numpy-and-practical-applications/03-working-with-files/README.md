# Part 3, Topic 3: Working with Files

Persisting your NumPy arrays to disk and loading them back into memory is a critical part of any data analysis workflow. NumPy provides simple and efficient functions for saving and loading arrays in various formats.

---

## 1. Saving and Loading Single Arrays in Binary Format

For saving a single `ndarray` object, the `np.save()` and `np.load()` functions are the most efficient choice. They use a custom, compressed binary format (`.npy`) that preserves the array's shape, data type, and other metadata.

### `np.save()`
Saves a single array to a `.npy` file.

```python
import numpy as np

# Create an array
arr = np.arange(24).reshape(4, 6)

# Save it to a file
np.save('my_array.npy', arr)
```
This creates a file named `my_array.npy` in the current directory.

### `np.load()`
Loads an array from a `.npy` file.

```python
# Load the array back from the file
loaded_arr = np.load('my_array.npy')

# The loaded array is identical to the original
print(np.array_equal(arr, loaded_arr)) # -> True
```

---

## 2. Saving and Loading Multiple Arrays

If you need to save multiple arrays into a single file, you can use `np.savez()`. This function saves the arrays into an uncompressed `.npz` archive.

### `np.savez()`
You pass the arrays as keyword arguments, and these keywords are used to identify the arrays when loading.

```python
# Create two arrays
a = np.array([[1, 2], [3, 4]])
b = np.arange(10)

# Save them into an archive
np.savez('my_archive.npz', array_a=a, array_b=b)
```

### Loading from an `.npz` file
When you load an `.npz` file, you get back a dictionary-like object. You can access the individual arrays by the keywords you used when saving.

```python
# Load the archive
archive = np.load('my_archive.npz')

# Access the arrays by their keys
print(archive['array_a'])
print(archive['array_b'])
```

For a compressed archive, you can use `np.savez_compressed()`, which works identically but produces a smaller file at the cost of slightly slower save/load times.

---

## 3. Working with Text Files

While binary formats are efficient, sometimes you need to work with human-readable text files, like CSVs (Comma-Separated Values).

### `np.savetxt()`
Saves an array to a text file. This is useful for exporting data to be used by other programs that can read delimited text.

```python
arr_2d = np.arange(1, 10).reshape(3, 3)

# Save the array to a CSV file
# fmt='%.2f' formats floats to 2 decimal places
# delimiter=',' sets the character to separate values
np.savetxt('my_data.csv', arr_2d, fmt='%d', delimiter=',')
```

### `np.loadtxt()`
Loads data from a text file. This function is fast but has some limitations (e.g., it assumes all data has the same type and cannot easily handle missing values).

```python
# Load the data from the CSV
loaded_from_text = np.loadtxt('my_data.csv', delimiter=',')

print(loaded_from_text)
```
For more complex text file parsing (e.g., files with headers, mixed data types, or missing values), it is highly recommended to use the **Pandas** library, specifically the `pandas.read_csv()` function, which is built on top of NumPy and designed for robust data ingestion.
