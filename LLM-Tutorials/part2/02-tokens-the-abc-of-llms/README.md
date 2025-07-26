# Part 2: How Language Models Actually Work
## Topic 2: Tokens - The ABCs of LLMs

We've established that LLMs are masters of predicting the "next word." But that's a slight simplification. In reality, LLMs don't see words. They see **tokens**.

Understanding tokens is fundamental to understanding how LLMs "think" and why they behave the way they do.

---

### What is a Token?

A computer can't read the word "apple" directly. It needs to convert it into a number. The most basic way to do this would be to assign a number to each letter (A=1, B=2, etc.), but that loses the meaning of the words. Another way would be to assign a unique number to every single word in the dictionary, but that would be inefficient for handling variations like "run," "running," and "ran."

Tokens are a clever compromise.

> **Simple Definition:** A token is a chunk of text that the model treats as a single unit. A token can be a whole word, a part of a word, a single character, or a punctuation mark.

An LLM has a fixed vocabulary of tokens it knows. This vocabulary is created by analyzing a massive amount of text and finding the most common and efficient ways to break it down.

**Let's look at an example sentence:**
`LLMs are powerful.`

A tokenizer might break this sentence down into the following tokens:
*   `L`
*   `L`
*   `Ms`
*   ` are`
*   ` powerful`
*   `.`

Notice a few things:
*   `LLMs` was broken into three parts: `L`, `L`, and `Ms`. This is because "LLMs" might not have been a common word in the training data, so the tokenizer breaks it down into smaller, more familiar pieces.
*   ` are` and ` powerful` have a space at the beginning. The tokenizer is smart enough to include the space as part of the token.
*   The period `.` is its own token.

**The Rule of Thumb:**
A good approximation is that **one token is roughly ¾ of a word** in English. So, 100 tokens is about 75 words.

### Why Use Tokens Instead of Words?

Using tokens provides a powerful balance between efficiency and meaning.

1.  **Handles Any Word:** Even if the model has never seen the word "technobabble" before, it can still process it by breaking it down into familiar tokens like `techno` and `babble`. This allows the model to handle misspellings, jargon, and new words.

2.  **Captures Word Relationships:** The model learns that the tokens `run`, `running`, and `ran` are related because they share the common root token `run`. This is much more efficient than treating them as three completely separate words.

3.  **Manages Vocabulary Size:** Instead of needing a dictionary with millions of words (including every possible variation and typo), the model can use a much smaller, fixed-size vocabulary of tokens (e.g., 50,000 to 100,000 tokens) to represent any possible text.

### Tokens and Model "Cost"

The number of tokens in your input and the model's output is a crucial concept because it's how AI companies measure usage and charge for their services.

*   **Context Window:** Every model has a maximum number of tokens it can handle at one time. This is called the **context window**. For example, a model with a 4,000-token context window can read your prompt and generate a response as long as the total number of tokens for both is less than 4,000. If you give it a longer text, it will forget the beginning.
*   **Pricing:** When you use an AI service through an API (which we'll cover later), you are typically billed per 1,000 tokens of input and per 1,000 tokens of output. A long, complex conversation costs more than a short, simple one.

In the next lesson, we'll explore what the model does with these tokens once it has them: it converts them into **embeddings**, the secret language of meaning.
