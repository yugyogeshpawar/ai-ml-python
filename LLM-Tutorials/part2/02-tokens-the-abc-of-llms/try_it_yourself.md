# Try It Yourself: Seeing How a Model "Sees" Text

This exercise will make the concept of tokens concrete. You will use an online tool to see exactly how different sentences are broken down into tokens by a real LLM.

---

### Exercise: The Online Tokenizer

For this exercise, we will use OpenAI's "Tokenizer" tool, which is used for their GPT models. It's a great way to build an intuition for how tokenization works in practice.

1.  **Open the Tokenizer tool in your browser:**
    *   **Click here:** [https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer)

2.  You will see a text box. Type or paste sentences into this box, and the tool will instantly show you how they are tokenized. The different colors represent the different tokens.

3.  **Start with a simple sentence.** Type the following and observe the output:
    > The quick brown fox jumps over the lazy dog.

    *   Look at the "Tokens" count below the box. How many tokens is this sentence?
    *   Notice how each word is its own token, and each token includes the preceding space.

4.  **Try a sentence with punctuation.**
    > Hello, world! This is a test.

    *   How are the punctuation marks (`,`, `!`, `.`) handled? Are they separate tokens?

5.  **Use a complex or unusual word.**
    > I love learning about Large Language Models!

    *   Look closely at how `Models!` is tokenized. Is it one token or multiple? Why do you think that is?
    *   Now try this: `Tokenization is fascinating.`
    *   How is the word `Tokenization` broken down? You can likely see the root word `Token` as one of the tokens.

6.  **Explore different languages.** If you know another language, try typing a sentence in it.
    > ¿Cómo estás?

    *   Is it more or less "token-efficient" than English? Languages that are less common in the training data often use more tokens to represent the same idea.

7.  **Try some "weird" text.**
    > `https://www.example.com/path?query=123`
    > `for i in range(10):`
    > `ThisIsAStrangeSentenceWithNoSpaces`

    *   Observe how the tokenizer tries its best to break down these unfamiliar structures into its known token vocabulary. This is how LLMs can process URLs, code, and other non-standard text formats.

**Reflection:**
After playing with the tokenizer, you should have a much better feel for what tokens are. You can see that it's not as simple as "one word = one token." This understanding is key to grasping why some prompts might be more "expensive" than others and how the model can handle such a wide variety of text.
