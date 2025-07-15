# random_number_generation_example.py

import numpy as np

# --- Best Practice: Create a Generator instance ---
# Using a seed ensures that the "random" numbers are the same every time
# the script is run, which is crucial for reproducibility.
rng = np.random.default_rng(seed=42)

# --- 1. Generating Random Floats and Integers ---
print("--- Floats and Integers ---")
# A single float between 0.0 and 1.0
print("Single random float:", rng.random())

# A 2x4 array of floats
print("\n2x4 array of random floats:\n", rng.random((2, 4)))

# A single integer between 0 and 9
print("\nSingle random integer:", rng.integers(10))

# A 1D array of 8 integers between 50 and 99
print("\nArray of 8 integers from 50 to 99:", rng.integers(50, 100, size=8))
print("-" * 30)


# --- 2. Sampling from Distributions ---
print("\n--- Sampling from Distributions ---")
# Generate a 5x2 array from the standard normal distribution (mean=0, std=1)
normal_samples = rng.standard_normal(size=(5, 2))
print("Samples from standard normal distribution:\n", normal_samples)

# Generate samples from a normal distribution with a specific mean and std dev
mu, sigma = 100, 15 # mean and standard deviation
iq_scores = rng.normal(mu, sigma, size=10)
print("\nSimulated IQ scores (mean=100, std=15):\n", iq_scores.astype(int))

# Generate samples from a uniform distribution
uniform_samples = rng.uniform(low=1, high=10, size=5)
print("\n5 samples from a uniform distribution between 1 and 10:\n", uniform_samples)
print("-" * 30)


# --- 3. Shuffling and Permutations ---
print("\n--- Shuffling and Permutations ---")
deck = np.arange(52) # A deck of cards represented by numbers 0-51
print("Original deck:", deck)

# Shuffle the deck in-place
rng.shuffle(deck)
print("Shuffled deck:", deck)

# To get a shuffled copy without modifying the original, use permutation()
original_deck = np.arange(10)
shuffled_copy = rng.permutation(original_deck)
print("\nOriginal small deck:", original_deck)
print("Shuffled copy (permutation):", shuffled_copy)
print("-" * 30)


# --- 4. Making Choices ---
print("\n--- Making Choices ---")
contestants = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
print("Contestants:", contestants)

# Select one winner
winner = rng.choice(contestants)
print("\nAnd the winner is:", winner)

# Select 3 contestants for a team (no replacement)
team = rng.choice(contestants, size=3, replace=False)
print("Team members (no replacement):", team)

# Simulate rolling a die 10 times (with replacement)
die_rolls = rng.choice(np.arange(1, 7), size=10, replace=True)
print("10 die rolls:", die_rolls)
print("-" * 30)
