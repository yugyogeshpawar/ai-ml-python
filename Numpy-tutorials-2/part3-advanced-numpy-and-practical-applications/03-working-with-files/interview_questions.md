# Interview Questions: Working with Files

---

### Question 1: What is the difference between `np.save` and `np.savez`? When would you use each?

**Answer:**

The difference lies in the number of arrays they are designed to handle:

-   **`np.save('filename.npy', arr)`**: This function is used to save a **single** NumPy array to a binary file with a `.npy` extension. It's a straightforward way to persist one array.

-   **`np.savez('filename.npz', name1=arr1, name2=arr2)`**: This function is used to save **multiple** NumPy arrays into a single archive file with a `.npz` extension. The arrays are saved as a dictionary-like structure, where you provide keyword arguments (`name1`, `name2`) that become the keys for accessing the arrays upon loading.

**When to use each:**
-   Use `np.save` when you have only one array to save. It's simple and direct.
-   Use `np.savez` (or `np.savez_compressed`) when you need to bundle several related arrays together into one file, such as when saving the weights and biases of a neural network model.

---

### Question 2: Why are NumPy's binary formats (`.npy`, `.npz`) generally preferred over text formats like CSV for saving NumPy arrays?

**Answer:**

NumPy's binary formats are preferred for several key reasons:

1.  **Efficiency:** Binary files are significantly **smaller** on disk and **faster** to read and write compared to text files. Text files store numbers as sequences of characters (e.g., "3.14159"), which takes up much more space than storing the number in its native binary representation (e.g., a 4- or 8-byte float).
2.  **Precision:** Binary formats store the numbers in their exact binary representation, meaning there is no loss of floating-point precision. When saving to a text file, you often have to specify a format, which can lead to rounding and loss of precision.
3.  **Metadata Preservation:** The `.npy` and `.npz` formats automatically save all the important metadata about the array, including its `shape`, `dtype`, and memory layout (`strides`). When you load the array with `np.load()`, it is perfectly reconstructed without any extra effort. A text file only stores the data values, losing this structural information.

Text formats like CSV should only be used when you need the data to be human-readable or when you need to export it to a non-NumPy application that cannot read binary files.

---

### Question 3: You need to load a large CSV file that contains columns with different data types (e.g., numbers and strings) and has a header row. Is `np.loadtxt()` a good choice for this task? If not, what tool would you recommend instead?

**Answer:**

No, **`np.loadtxt()` is not a good choice** for this task.

`np.loadtxt()` is a simple and fast function, but it has significant limitations:
-   It expects all data in the file to be of the same type (e.g., all floats or all integers). It cannot handle mixed data types in different columns.
-   It does not have a simple way to handle missing values.
-   While it can skip a header row (`skiprows=1`), it is not designed for robust parsing of complex file structures.

**Recommended Tool:**
The highly recommended tool for this job is the **Pandas library**, specifically the **`pandas.read_csv()`** function.

Pandas is built on top of NumPy and is the de facto standard for data analysis in Python. `pd.read_csv()` is designed to be powerful and flexible, easily handling:
-   Mixed data types across columns.
-   Header rows (it automatically detects them).
-   Missing data (it can represent them as `NaN`).
-   Different delimiters, encodings, and much more.

It reads the data into a Pandas DataFrame, which is a 2D labeled data structure with columns of potentially different types—perfect for real-world datasets.
