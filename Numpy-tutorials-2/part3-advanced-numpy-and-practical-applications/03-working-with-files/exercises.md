# Exercises: Working with Files

These exercises will help you practice saving and loading NumPy arrays in different formats.

---

### Exercise 1: Saving Model Weights

**Task:**

Imagine you have trained a simple neural network. The "weights" of this network are stored in several NumPy arrays. You need to save these weights to a single file so you can load them later for inference.

1.  Create three separate NumPy arrays to represent the weights and biases of two layers:
    -   `layer1_weights` (shape 10x5)
    -   `layer1_biases` (shape 5,)
    -   `layer2_weights` (shape 5x2)
    -   `layer2_biases` (shape 2,)
    Fill them with random numbers.
2.  Use `np.savez_compressed()` to save all four arrays into a single compressed file named `model_weights.npz`. Use descriptive keywords for each array (e.g., `l1_w`, `l1_b`, etc.).
3.  Write a separate piece of code that loads this `.npz` file.
4.  Print the shape of each loaded array to verify that the data was saved and loaded correctly.

---

### Exercise 2: Appending to a Log File

**Task:**

You are running a simulation and want to log the results (e.g., a 1x3 vector of metrics) at the end of each step. You want to save these results to a human-readable text file.

1.  Create a 1x3 NumPy array representing the results of the first simulation step.
2.  Open a file named `simulation_log.csv` in **append mode** (`'ab'`).
3.  Use `np.savetxt()` to append the results to the file. You will need to pass the file handle to `savetxt` instead of a filename.
4.  Repeat steps 1-3 inside a loop (e.g., 5 times) to simulate multiple steps. Make sure to close the file handle after the loop.
5.  After the loop finishes, use `np.loadtxt()` to load the entire `simulation_log.csv` file back into a single NumPy array.
6.  Print the final array and its shape to confirm that all steps were logged correctly.

**Note:** Opening the file in append mode is the key here. If you just pass the filename to `savetxt` in a loop, it will overwrite the file every time.

---

### Exercise 3: Binary vs. Text File Size Comparison

**Task:**

This exercise will demonstrate the significant difference in file size between NumPy's binary format (`.npy`) and a standard text format (`.csv`).

1.  Create a large 1000x50 array of random floating-point numbers.
2.  Save this array to a file named `large_array.npy` using `np.save()`.
3.  Save the same array to a file named `large_array.csv` using `np.savetxt()`. You can use the default formatting.
4.  Use Python's `os` module to get the file size of both files in bytes (`os.path.getsize('filename')`).
5.  Print the file sizes for both the `.npy` and `.csv` files.
6.  Calculate and print how many times larger the text file is compared to the binary file. This illustrates why `.npy` is preferred for large numerical datasets.
