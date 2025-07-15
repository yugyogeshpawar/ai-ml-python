# Part 3, Topic 2: Random Number Generation

Generating random numbers is essential for a wide range of applications, including simulations, statistical sampling, and initializing weights in machine learning models. NumPy's `np.random` module provides a powerful and flexible suite of tools for this purpose.

**Modern Approach:** Since NumPy version 1.17, the recommended best practice is to use a `Generator` instance. You create a generator and then call its methods to produce random numbers. This approach offers better reproducibility and statistical properties than the older legacy methods.

```python
import numpy as np

# Create a default random number generator
rng = np.random.default_rng(seed=42) # Using a seed makes the results reproducible
```

---

## 1. Generating Random Floats and Integers

### `Generator.random()`
Generates random floating-point numbers in the half-open interval `[0.0, 1.0)`.

```python
# Generate a single random float
print(rng.random())

# Generate a 1D array of 5 random floats
print(rng.random(5))

# Generate a 2x3 array of random floats
print(rng.random((2, 3)))
```

### `Generator.integers()`
Generates random integers within a specified range.

**Syntax:** `rng.integers(low, high=None, size=None, endpoint=False)`

-   If `high` is `None`, the range is `[0, low)`.
-   If `high` is provided, the range is `[low, high)`.
-   Set `endpoint=True` to make the `high` value inclusive `[low, high]`.

```python
# Generate a single integer between 0 and 9
print(rng.integers(10))

# Generate a 1D array of 5 integers between 10 and 19
print(rng.integers(10, 20, size=5))

# Generate a 3x4 array of integers between 0 and 100 (inclusive)
print(rng.integers(0, 101, size=(3, 4), endpoint=True))
```

---

## 2. Sampling from Standard Distributions

The `Generator` object can also draw samples from many common probability distributions.

### `Generator.standard_normal()`
Draws samples from a standard normal distribution (mean=0, standard deviation=1). This is often called "Gaussian" or "bell curve" distribution.

```python
# Generate a 3x3 array of normally distributed values
normal_samples = rng.standard_normal(size=(3, 3))
print(normal_samples)

# You can verify the mean and std deviation
print(f"Mean: {normal_samples.mean():.2f}") # Should be close to 0
print(f"Std Dev: {normal_samples.std():.2f}") # Should be close to 1
```

### `Generator.uniform()`
Draws samples from a uniform distribution, where every value in the given range is equally likely.

```python
# Generate a 1D array of 5 floats between -5.0 and 5.0
uniform_samples = rng.uniform(low=-5, high=5, size=5)
print(uniform_samples)
```

---

## 3. Shuffling and Choosing

### `Generator.shuffle()`
Modifies a sequence by shuffling its contents **in-place**.

```python
arr = np.arange(10)
print("Original array:", arr)
rng.shuffle(arr)
print("Shuffled array:", arr)
```

### `Generator.choice()`
Randomly chooses a specified number of items from an array, with or without replacement.

```python
elements = np.array(['apple', 'banana', 'cherry', 'date'])

# Choose 3 elements without replacement (default)
print(rng.choice(elements, size=3, replace=False))

# Choose 10 elements with replacement
print(rng.choice(elements, size=10, replace=True))
```

Using a seeded `Generator` is the key to writing reproducible, high-quality simulations and experiments in NumPy.
