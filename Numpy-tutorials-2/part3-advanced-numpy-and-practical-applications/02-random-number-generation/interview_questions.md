# Interview Questions: Random Number Generation

---

### Question 1: What is the modern, recommended way to generate random numbers in NumPy, and why is it preferred over older methods like `np.random.rand()`?

**Answer:**

The modern, recommended way is to first create an instance of a **random number generator** and then call methods on that instance. This is done using `rng = np.random.default_rng()`.

This approach is preferred for several key reasons:

1.  **Reproducibility and Control:** The generator object (`rng`) holds the state of the random number generation. By creating a generator with a specific `seed` (e.g., `np.random.default_rng(seed=42)`), you guarantee that the sequence of random numbers will be exactly the same every time the code is run. This is crucial for debugging, testing, and creating reproducible scientific experiments. The older methods use a global state, which can be unpredictably affected by other parts of the code or libraries.
2.  **Better Statistical Properties:** The algorithms used by the new `Generator` system are more modern and have better statistical properties than the legacy ones, ensuring a higher quality of randomness.
3.  **No Global State:** The legacy functions (`np.random.rand()`, `np.random.randn()`, etc.) modify a single, global random state. This can cause issues in complex applications where different components might interfere with each other's random sequences. Using explicit generator instances avoids this problem entirely.

---

### Question 2: Explain the purpose of setting a "seed" when creating a random number generator.

**Answer:**

A "seed" is an initial value used to start a sequence of pseudo-random numbers. A pseudo-random number generator is an algorithm that produces a sequence of numbers that appears random, but is in fact completely determined by the seed value.

The purpose of setting a seed is **reproducibility**.

-   If you use the same seed, you will get the exact same sequence of "random" numbers every single time you run the program.
-   If you do not provide a seed, the generator is typically seeded from a source of true randomness from the operating system (like the current time or system entropy), which means you will get a different sequence of numbers on each run.

In scientific computing and machine learning, setting a seed is essential for debugging code, sharing work with others, and ensuring that experiments can be replicated to produce the same results.

---

### Question 3: What is the difference between `rng.shuffle()` and `rng.permutation()`?

**Answer:**

Both functions are used to reorder the elements of an array, but they differ in one critical way: whether they modify the original array or return a new one.

-   **`rng.shuffle(arr)`**: This function shuffles the elements of the input array `arr` **in-place**. It modifies the original array directly and returns `None`.

-   **`rng.permutation(arr)`**: This function returns a **new, shuffled copy** of the array. The original array `arr` is left unchanged. It can also be called with an integer `N`, in which case it returns a shuffled permutation of `np.arange(N)`.

**When to use which:**
-   Use `shuffle()` when you are happy to modify the original array and want to save memory by not creating a new one.
-   Use `permutation()` when you need to keep the original array intact and work with a shuffled version of it.
