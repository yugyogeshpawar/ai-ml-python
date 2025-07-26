# Deep Dive: From Word to Sentence Embeddings

**Note:** This optional section discusses how models go from understanding single tokens to understanding whole sentences.

---

So far, we've discussed how an LLM has an embedding vector for each *token* in its vocabulary. But when you use a tool like the sentence similarity demo, you're not getting the embedding for a single token; you're getting a single embedding vector for the *entire sentence*.

How does the model combine the individual token embeddings into one sentence embedding? This is a crucial step, and there are several ways to do it.

### 1. Simple Averaging (The "Bag of Words" Approach)

The simplest method is to just take the embedding vectors of all the tokens in the sentence and average them together.

*   **How it works:** You get the vector for "The," the vector for "cat," the vector for "sat," etc., and you average all the numbers in each position of the vector. The result is a new vector of the same dimension that represents the "average meaning" of the sentence.
*   **Pros:** It's very fast and simple.
*   **Cons:** It's a "bag of words" approach. It loses all information about word order and grammar. The sentences "The dog chased the cat" and "The cat chased the dog" would have the exact same sentence embedding, even though their meanings are completely different. For many simple applications (like identifying the general topic of a document), this is good enough. For chatbots and more complex tasks, it's not.

### 2. Using Special Tokens (like the `[CLS]` token)

Models like BERT (a foundational model from Google) came up with a clever solution. They add a special token, called `[CLS]` (for "classification"), to the beginning of every input sentence.

*   **How it works:** The sentence `The cat sat on the mat` becomes `[CLS] The cat sat on the mat`.
*   This whole sequence of tokens is then processed through the deep layers of the Transformer network (which we'll cover next).
*   The Transformer's "attention" mechanism is designed to make tokens "talk to each other" and understand the context.
*   By the time the sequence gets to the final layer of the model, the embedding for the special `[CLS]` token has been modified to represent a summary of the entire sentence's meaning, including word order and context.
*   For sentence similarity tasks, developers simply use the final-state embedding of the `[CLS]` token as the embedding for the whole sentence.

### 3. Pooling Strategies

Modern sentence-embedding models often use a "pooling" strategy, which is a more sophisticated way of combining the final token embeddings.

After all the tokens have passed through the entire Transformer network and have been "contextualized" (meaning their vectors have been updated based on the other words in the sentence), a pooling layer combines them.

*   **Mean Pooling:** This is the most common strategy. It's similar to the simple averaging we discussed first, but it's performed on the *final, contextualized* token embeddings, not the initial ones. This means the word order and context information from the Transformer is preserved before the averaging happens. The sentence similarity demo you used likely uses this method.
*   **Max Pooling:** Instead of averaging, this strategy takes the maximum value for each position in the vector across all tokens. This can sometimes be better at capturing the single most important "signal" or feature in the sentence.

The choice of which strategy to use depends on the specific task the model is being fine-tuned for, but Mean Pooling is a very strong and common baseline.
