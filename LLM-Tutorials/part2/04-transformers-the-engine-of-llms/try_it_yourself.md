# Try It Yourself: Visualizing Attention

It's hard to truly visualize what's happening inside a giant neural network, but some clever tools can give us a glimpse of the attention mechanism in action.

---

### Exercise: The Attention Visualizer

For this exercise, we'll use a tool that shows the "attention heads" of a real Transformer model (BERT, from Google) as it processes a sentence.

1.  **Open the "exBERT" visualization tool.**
    *   **Click here:** [https://huggingface.co/spaces/exbert-team/exbert](https://huggingface.co/spaces/exbert-team/exbert)
    *   This tool may take a moment to load.

2.  **Select a sentence.** On the left-hand side, under "Select a sample," you can choose from several pre-written sentences. Let's stick with the default:
    > The cat sat on the mat.

3.  **Look at the "Attention" visualization.** In the main panel, you will see a visualization of the attention patterns. It might look complex, but we can simplify it.
    *   The lines connect the token being processed (on the left) to the tokens it is "attending to" (on the right).
    *   The thickness of the line represents the strength of the attention score.

4.  **Focus on a single word.** On the left-hand list of tokens, hover your mouse over the word **"sat"**.
    *   The visualization will update to show only the attention patterns for the word "sat."
    *   You will likely see strong connections from "sat" to "cat" and "mat." The model has learned that the thing doing the "sitting" is the "cat," and the place it is sitting is the "mat."

5.  **Test pronoun resolution.**
    *   In the text box at the top, type in a new sentence with a pronoun:
        > The dog chased the ball because it was fast.
    *   Click **"Submit"**.
    *   Now, on the left, hover over the token for **"it"**.
    *   Where are the strongest attention lines pointing? They should be pointing clearly to **"dog"**, not "ball." The model correctly understands what "it" refers to.

6.  **Try to trick it!**
    *   Change the sentence to:
        > The dog chased the ball because it was red.
    *   Click **"Submit"**.
    *   Now, hover over **"it"** again.
    *   Where does the attention point now? It should point directly to **"ball"**.

**Reflection:**
By playing with this tool, you are getting a direct, visual confirmation of the "cocktail party" analogy. For every word, the model is dynamically deciding which other words are most important for understanding its meaning in context. This ability to form long-range connections is the core reason why Transformers are so powerful.
