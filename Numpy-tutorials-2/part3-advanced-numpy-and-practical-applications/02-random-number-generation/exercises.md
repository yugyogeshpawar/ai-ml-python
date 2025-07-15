# Exercises: Random Number Generation

These exercises will help you practice using NumPy's modern random number generation tools for various simulation tasks.

---

### Exercise 1: Simulating a Biased Coin

**Task:**

Simulate 10,000 flips of a biased coin that lands on "Heads" 70% of the time and "Tails" 30% of the time.

1.  Create a random number generator with a fixed seed for reproducibility.
2.  Use the `rng.choice()` method to perform the simulation.
    -   The choices are `['Heads', 'Tails']`.
    -   You need to specify the probabilities for each choice using the `p` argument.
    -   The size of the simulation should be 10,000.
3.  Count the number of "Heads" and "Tails" in your result.
4.  Calculate the percentage of "Heads" and "Tails" and print them to verify that they are close to the 70/30 split.

---

### Exercise 2: Creating a Test Dataset

**Task:**

Generate a synthetic dataset of 1,000 "people" with the following three features:
-   `height`: Normally distributed with a mean of 175 cm and a standard deviation of 10 cm.
-   `weight`: Normally distributed with a mean of 70 kg and a standard deviation of 15 kg.
-   `age`: Uniformly distributed integers between 18 and 65 (inclusive).

1.  Create a seeded random number generator.
2.  Generate three separate 1D arrays for `height`, `weight`, and `age` using the appropriate generator methods (`rng.normal`, `rng.integers`).
3.  Combine these three arrays into a single 1000x3 NumPy array. You can use `np.column_stack()` for this.
4.  Print the shape of the final dataset and the first 5 rows to see the result.

---

### Exercise 3: Bootstrap Sampling

**Task:**

Bootstrapping is a statistical method that involves resampling a dataset with replacement to estimate properties of a population.

1.  Create a 1D NumPy array representing a small sample of 20 data points. You can generate this with `rng.integers(0, 100, size=20)`.
2.  Calculate the mean of this original sample.
3.  Now, perform a bootstrap:
    -   Use `rng.choice()` to create a new sample of the *same size* (20) by drawing from your original sample **with replacement**.
    -   Calculate the mean of this new bootstrap sample.
4.  Repeat the process in step 3 in a loop (e.g., 1,000 times), storing the mean of each bootstrap sample in a list.
5.  You will now have a list of 1,000 bootstrap means. Calculate the standard deviation of this list. This value is the "bootstrap standard error" of the mean, which gives you an idea of the uncertainty in your original sample mean.
