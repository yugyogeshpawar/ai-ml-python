# Try It Yourself: Feeling the "Semantic Distance"

This exercise is designed to give you a hands-on feel for how embeddings place concepts on a map. You will compare the "semantic distance" between different sentences.

---

### Exercise: Compare Sentence Embeddings

For this exercise, we'll use a Hugging Face Space that lets you type in a sentence and get its embedding vector. More importantly, it lets you compare two sentences and see how "similar" the model thinks they are.

1.  **Open the Sentence Similarity tool:**
    *   **Click here:** [https://huggingface.co/spaces/sentence-transformers/sentence-similarity](https://huggingface.co/spaces/sentence-transformers/sentence-similarity)

2.  You will see a "Source Sentence" and several other text boxes below it. The tool calculates a **similarity score** between the source sentence and each of the other sentences. The score ranges from 0 (completely unrelated) to 1 (identical meaning).

3.  **Test #1: High Similarity.**
    *   In the **Source Sentence** box, type:
        > A man is eating a piece of bread.
    *   In the first box below it, type:
        > A guy is consuming some food.
    *   Click the **"Compute"** button.
    *   **Observe the score.** It should be very high (likely above 0.8). Even though the sentences use different words, the model's embeddings capture the fact that their underlying meaning is almost the same.

4.  **Test #2: Medium Similarity.**
    *   Keep the same source sentence: `A man is eating a piece of bread.`
    *   In the second box, type something that is related but different:
        > A woman is at a restaurant.
    *   Click **"Compute"**.
    *   **Observe the score.** It should be much lower than the first test, but still not zero. The model recognizes that "man/woman," "eating/restaurant," and "bread/food" are all in the same general "food" neighborhood on its map of meaning, but it knows they are not the same.

5.  **Test #3: Low Similarity.**
    *   Keep the same source sentence: `A man is eating a piece of bread.`
    *   In the third box, type something completely unrelated:
        > The rocket is launching into space.
    *   Click **"Compute"**.
    *   **Observe the score.** It should be very low (likely close to 0). The model knows that the concepts in this sentence are in a totally different part of the "meaning map" from the concepts in the source sentence.

6.  **Play around!** Try your own sentences.
    *   How similar are "What is the weather today?" and "Will I need an umbrella?"
    *   Compare a positive movie review to a negative one.
    *   Write a short paragraph and then write a summary of it. How similar are they?

**Reflection:**
This tool gives you a direct window into how an AI model perceives meaning. The "similarity score" is a direct calculation of the distance between the embedding vectors of the sentences. By playing with it, you can build a strong intuition for the "geometry of meaning" that powers modern AI.
