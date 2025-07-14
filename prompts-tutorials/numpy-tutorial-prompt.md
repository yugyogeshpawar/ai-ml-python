# Prompt for Generating a Comprehensive NumPy Tutorial

## 1. Overall Objective
Generate a complete, multi-part NumPy tutorial suitable for hosting on GitHub. The tutorial should provide a deep dive into the NumPy library, covering its core features and applications in scientific computing and data analysis.

## 2. Target Audience
The tutorial is for Python developers, data scientists, engineers, and students who want to master numerical computing in Python. No prior experience with NumPy is required, but a basic understanding of Python is assumed.

## 3. Core Philosophy & Style
- **Foundation First:** Emphasize that NumPy is the fundamental package for scientific computing in Python, forming the basis for many other libraries like Pandas, SciPy, and Scikit-learn.
- **Efficiency and Vectorization:** Clearly explain the performance benefits of using vectorized NumPy operations over standard Python loops.
- **Practical and Problem-Oriented:** Ground every lesson in practical examples that solve common numerical problems.

## 4. High-Level Structure
The tutorial will be divided into three main parts.

- **Part 1: NumPy Fundamentals**
- **Part 2: Array Manipulation and Operations**
- **Part 3: Advanced NumPy and Practical Applications**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following four files inside a dedicated topic directory (e.g., `part1-fundamentals/01-introduction-to-numpy/`):

1.  **`README.md`**: The main lesson file with detailed explanations.
2.  **`[topic]_example.py`**: A runnable Python script demonstrating the lesson's concept.
3.  **`exercises.md`**: A file containing 2-3 practical exercises.
4.  **`interview_questions.md`**: A file with 3 relevant interview questions and detailed answers.

---

### **Part 1: NumPy Fundamentals**
- **Goal:** Introduce the core NumPy `ndarray` object and its essential attributes.
- **Topics:**
    1.  `01-introduction-to-numpy`: What is NumPy? Why use it? The concept of vectorization.
    2.  `02-creating-numpy-arrays`: Creating arrays from lists, using `np.zeros`, `np.ones`, `np.arange`, and `np.linspace`.
    3.  `03-array-attributes-and-data-types`: Understanding `shape`, `ndim`, `size`, `dtype`, and how to specify data types.
    4.  `04-basic-indexing-and-slicing`: Accessing and slicing 1D, 2D, and 3D arrays.

- **Project:** A script that creates and inspects various NumPy arrays, reporting their shape, data type, and dimensions.

### **Part 2: Array Manipulation and Operations**
- **Goal:** Cover the rich set of functions for manipulating and performing computations on arrays.
- **Topics:**
    1.  `01-universal-functions-ufuncs`: Element-wise operations (e.g., `np.add`, `np.sqrt`, `np.exp`).
    2.  `02-array-broadcasting`: The powerful mechanism that allows NumPy to perform operations on arrays of different shapes.
    3.  `03-advanced-indexing`: Boolean array indexing and integer array indexing (fancy indexing).
    4.  `04-statistical-methods`: Using aggregate functions like `np.sum`, `np.mean`, `np.std`, `np.min`, and `np.max` along different axes.
    5.  `05-reshaping-and-transposing`: Changing the shape of arrays with `reshape`, `ravel`, and the `.T` attribute.

- **Project:** A data analysis script that loads a numeric dataset (e.g., from a CSV file), cleans it by removing outliers using boolean indexing, and calculates summary statistics.

### **Part 3: Advanced NumPy and Practical Applications**
- **Goal:** Explore more advanced features and demonstrate how NumPy is used in real-world scenarios.
- **Topics:**
    1.  `01-linear-algebra-with-numpy`: The `np.linalg` module for matrix multiplication (`@` operator), finding the determinant, inverse, and eigenvalues.
    2.  `02-random-number-generation`: The `np.random` module for generating random data from various distributions.
    3.  `03-working-with-files`: Saving and loading NumPy arrays to and from disk (`np.save`, `np.load`, `np.savetxt`).
    4.  `04-integrating-with-other-libraries`: A brief overview of how NumPy arrays are used as the primary data structure in libraries like Pandas and Matplotlib.

- **Project:** A simple image processing application that loads an image as a NumPy array, applies filters (e.g., grayscale, brightness adjustment), and saves the result.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Clearly explain the "why" behind each feature (e.g., why broadcasting is so powerful).
    - Use diagrams to illustrate complex concepts like broadcasting and reshaping.
- **For `[topic]_example.py` files:**
    - Code should be clean, runnable, and well-commented.
- **For `exercises.md` and `interview_questions.md` files:**
    - Focus on practical problems that require a solid understanding of NumPy's array manipulation capabilities.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
