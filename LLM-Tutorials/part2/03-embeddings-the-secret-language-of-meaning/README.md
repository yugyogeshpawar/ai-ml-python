# Part 2: How Language Models Actually Work
## Topic 3: Embeddings - The Secret Language of Meaning

We now know that an LLM breaks our text down into tokens. But a token is still just a label. The model needs a way to understand the *meaning* and *relationships* behind these tokens.

This is where the magic of **embeddings** comes in. This is one of the most important concepts in all of modern AI.

---

### From Tokens to Vectors

In Part 1, we introduced the idea of a "map of meaning," where concepts are represented by vectors (lists of numbers). An embedding is the process of taking a token and finding its corresponding vector on that map.

> **Simple Definition:** An embedding is a vector representation of a token. This vector captures the token's semantic meaning, allowing the model to understand its relationship to other tokens.

Think of it like a giant dictionary or lookup table inside the AI's brain.

1.  **Input:** The model receives a token, for example, the token for the word "cat" (let's say its ID is `5842`).
2.  **Lookup:** The model goes to its massive "embedding table" and looks up entry #5842.
3.  **Output:** The entry it finds is the **embedding vector** for "cat"—a list of hundreds or thousands of numbers (e.g., `[4.2, -1.5, 3.8, ..., 0.9]`).

This vector is the model's internal, mathematical representation of what a "cat" is. Every single token in the model's vocabulary has its own unique embedding vector.

### The "Map of Meaning"

The power of embeddings is not in the numbers themselves, but in the **geometry of the embedding space**. As we discussed before, tokens with similar meanings will have vectors that are close to each other on this high-dimensional map.

*   The vector for "kitten" will be extremely close to the vector for "cat."
*   The vector for "puppy" will be close to "dog," and both will be in the same general neighborhood as "cat" and "kitten" (the "pets" neighborhood).
*   The vector for "boat" will be in a completely different region of the map, but it might be close to "ship" and "water."

This geometric relationship is what allows the model to reason about concepts. When it sees the token for "cat," it's not just seeing a word; it's activating a specific point in its "meaning map," which is located near all the other concepts it has learned are related to cats.

### How are Embeddings Learned?

This is the most amazing part. No human programs these meanings. The embeddings are **learned automatically** during the model's training process.

Remember the goal of training: to get better at predicting the next token.

Imagine the model is processing the sentence: "The fluffy cat sat on the..."

To get good at predicting the next word (`mat`), the model needs to learn that "fluffy" and "cat" are concepts that often appear together. To do this, the Gradient Descent algorithm (our "walking downhill" friend) will slightly nudge the embedding vectors for "fluffy" and "cat" closer to each other in the embedding space.

Now, imagine it sees millions of other sentences:
*   "The cute cat..." (Nudges "cute" and "cat" closer)
*   "The fluffy dog..." (Nudges "fluffy" and "dog" closer)
*   "My dog is cute." (Nudges "dog" and "cute" closer)

After processing trillions of words, a rich, complex map of meaning emerges naturally. The model has learned that "fluffy," "cute," "cat," and "dog" all belong in a similar region of the map, without ever being explicitly told what a "pet" is.

The embedding layer is the very first part of the LLM's architecture. It's the gateway through which all text must pass to be understood by the rest of the model's brain. In the next lesson, we'll look at the engine that processes these embeddings: the **Transformer**.
