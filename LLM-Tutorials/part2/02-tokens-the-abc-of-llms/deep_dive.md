# Deep Dive: Tokenization Algorithms

**Note:** This optional section briefly touches on the algorithms used to create the token vocabularies for LLMs.

---

How does a model decide what its tokens should be? It can't be done by hand. Instead, developers use special algorithms that learn the optimal way to break down text from a massive corpus of sample documents. The goal is to find a vocabulary that can represent any text efficiently.

The most common family of algorithms used for this is **Subword Tokenization**. The key idea is to keep common words as single tokens while breaking down rare words into smaller, meaningful subword units.

Here are two of the most popular subword tokenization algorithms:

### 1. Byte-Pair Encoding (BPE)

BPE is one of the most widely used tokenization algorithms, and it's the basis for the tokenizer used in OpenAI's GPT models.

**How it works (a simplified view):**

1.  **Start with characters:** The initial vocabulary consists of every single character in the text (e.g., `a, b, c, d, ...`). Every word is represented as a sequence of these character tokens. For example, `hug` is `h, u, g`.

2.  **Find the most frequent pair:** The algorithm then scans the entire text corpus and finds the pair of adjacent tokens that appears most frequently. For example, perhaps the pair `h` and `u` is very common.

3.  **Merge the pair:** It merges this most frequent pair into a new, single token. So, `hu` becomes a new token in the vocabulary. The word `hug` is now represented as `hu, g`.

4.  **Repeat:** The algorithm repeats this process over and over. Maybe the next most common pair is `t` and `h`. It merges them into `th`. Then maybe it finds that `th` and `e` are very common, so it merges them into `the`.

5.  **Set a vocabulary size:** This process is repeated for a predetermined number of merges, which defines the final vocabulary size (e.g., 50,000 merges for a vocabulary of 50,000 tokens).

**The result:**
*   Very common words (like `the`, `and`, `is`) quickly become single tokens.
*   Less common words (like `hugging`) might be represented as two tokens: `hugg` and `ing`.
*   Rare or new words (like `tokenization`) are represented by a sequence of subword tokens that the model has seen before: `token`, `iz`, `ation`.

### 2. WordPiece

WordPiece is a very similar algorithm used by Google for models like BERT and Gemini. The main difference is in how it decides which pair of tokens to merge.

*   Instead of just picking the most frequent pair, WordPiece calculates a likelihood score. It merges the pair that does the best job of maximizing the probability of the training data if it were tokenized with the new, merged vocabulary.

The practical result is very similar to BPE, but this different mathematical approach can sometimes lead to a slightly different and potentially more efficient set of tokens.

### 3. Unigram Language Model

This is another approach used by models like T5 and XLNet. Unlike BPE and WordPiece, which are additive (they build up the vocabulary), the Unigram model is subtractive.

1.  **Start with a huge vocabulary:** It starts with a very large number of potential tokens (e.g., all pre-tokenized words and common substrings).
2.  **Iteratively remove tokens:** It then uses a statistical model to calculate a "loss score" for each token, representing how much the overall efficiency would suffer if that token were removed from the vocabulary.
3.  **Prune the vocabulary:** It progressively removes the "worst" tokens—the ones that have the least negative impact—until the vocabulary reaches the desired size.

The end goal of all these algorithms is the same: to create a compact, efficient, and flexible vocabulary of tokens that allows the LLM to process any text it encounters.
