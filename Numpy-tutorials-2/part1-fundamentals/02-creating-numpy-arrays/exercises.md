# Exercises: Creating NumPy Arrays

These exercises will help you practice the different methods of array creation in NumPy.

---

### Exercise 1: Create a Checkerboard

**Task:**

Create an 8x8 NumPy array representing a checkerboard pattern. The board should be made of 0s and 1s.

**Hint:**

-   Start by creating an 8x8 array of zeros using `np.zeros()`.
-   Use slicing to select and fill in the `1`s. For example, `array[start:stop:step]` can be used to select every other element. You will need to do this for both rows and columns.
-   There are multiple ways to solve this. Can you find a solution that uses slicing and another that might use a mathematical function?

**Expected Output:**
```
[[0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]]
```

---

### Exercise 2: `arange` vs. `linspace`

**Task:**

Both `np.arange()` and `np.linspace()` can create sequences of numbers, but they serve different purposes. Your task is to highlight this difference.

1.  Create an array named `arr_arange` that contains all integer numbers from 10 to 50 (inclusive) using `np.arange()`.
2.  Create an array named `arr_linspace` that contains exactly 5 evenly spaced numbers between 10 and 50 (inclusive) using `np.linspace()`.
3.  Now, try to create the *same* array as `arr_linspace` but using `np.arange()`. What `step` value do you need? Is it as intuitive as using `linspace`?
4.  Print both arrays and write a short comment explaining a scenario where you would prefer `linspace` over `arange`.

---

### Exercise 3: Generating a Color Gradient

**Task:**

Imagine you are creating a color gradient for a visualization. A color can be represented by an RGB triplet (Red, Green, Blue), where each value is between 0 and 255.

You want to create a gradient that transitions smoothly from pure blue `(0, 0, 255)` to pure red `(255, 0, 0)`.

1.  Define the number of steps in the gradient, for example, `n_steps = 10`.
2.  Create a 2D NumPy array of shape `(n_steps, 3)` and data type `int` to hold the gradient. You can start with an array of zeros.
3.  Use `np.linspace()` to generate the values for the Red channel (from 0 to 255).
4.  Use `np.linspace()` to generate the values for the Blue channel (from 255 down to 0).
5.  Assign these generated sequences to the appropriate columns of your gradient array. The Green channel should remain 0.
6.  Print the final gradient array.
