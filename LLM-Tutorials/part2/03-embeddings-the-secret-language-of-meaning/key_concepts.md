# Key Concepts: Embeddings

Here are the most important terms from this lesson, explained in simple English.

### 1. Embedding
-   **What it is:** A vector (a list of numbers) that represents the meaning of a token.
-   **Analogy:** A single point on a giant, multi-dimensional "map of meaning." Every token in the model's vocabulary has a unique coordinate on this map.
-   **Why it matters:** This is the crucial step where text is translated into a format the AI can actually work with. Embeddings turn words into meaningful mathematical objects.

### 2. Embedding Space
-   **What it is:** The "map" itself. It's the vast, high-dimensional space where all the embedding vectors live.
-   **Analogy:** A galaxy where every star is a token. The magic is in the distances between the stars. Stars representing similar concepts (like "cat" and "kitten") are close together, forming constellations of meaning. Stars representing unrelated concepts (like "cat" and "car") are light-years apart.
-   **Why it matters:** The geometry of this space *is* the model's understanding of language. By navigating this space, the model can understand relationships, analogies, and context.

### 3. Semantic Similarity
-   **What it is:** The idea that things with similar meanings should have similar embedding vectors.
-   **Analogy:** "Birds of a feather flock together." In the embedding space, words that are used in similar contexts in the real world will be "flocked together" in the same neighborhood on the map.
-   **Why it matters:** This is the fundamental principle that makes embeddings useful. It allows us to perform "concept math" (like `King - Man + Woman = Queen`) and is the basis for many AI applications, including search engines, recommendation systems, and chatbots. When you search for "cars," a smart search engine can also show you results for "automobiles" because it knows their embedding vectors are very close.
