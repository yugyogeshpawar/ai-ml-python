# Examples in Action: Tokens

Seeing how different types of text are broken down into tokens is the best way to build an intuition for them. Let's look at some specific examples, imagining we're using a tokenizer like the one for GPT models.

### 1. Common English Words
Simple, common words are almost always a single token.

*   **Text:** `The cat sat on the mat.`
*   **Tokens:** `| The` `| cat` `| sat` `| on` `| the` `| mat` `|.`
*   **Analysis:** Each word is one token. The leading space is included with the word. The period is its own token. This is very efficient.

---

### 2. Less Common or Compound Words
Longer or less common words are often broken down into meaningful sub-word units.

*   **Text:** `Tokenization is fascinating.`
*   **Tokens:** `| Token` `|ization` `| is` `| fascin` `|ating` `|.`
*   **Analysis:**
    *   `Tokenization` is split into `Token` and `ization`. The model recognizes the common root word and the common suffix. This allows it to understand the relationship between "Token" and "Tokenization."
    *   `fascinating` is split into `fascin` and `ating`. Again, the model finds a common root and suffix.

---

### 3. Plurals and Verb Tenses
Tokens help the model understand grammatical variations efficiently.

*   **Text:** `The models are modeling.`
*   **Tokens:** `| The` `| models` `| are` `| modeling` `|.`
*   **Wait, let's look closer:** A real tokenizer would likely do this:
*   **Tokens (Realistic):** `| The` `| model` `|s` `| are` `| model` `|ing` `|.`
*   **Analysis:** The tokenizer breaks `models` into `model` + `s` and `modeling` into `model` + `ing`. This is incredibly powerful. The model now knows that all three words (`model`, `models`, `modeling`) share the same core concept (`model`). It learns the meaning of the `s` and `ing` tokens as grammatical modifiers.

---

### 4. Non-English Languages
Tokenization efficiency can vary by language, often depending on how much of that language was in the training data.

*   **Text (Spanish):** `¿Dónde está la biblioteca?`
*   **Tokens:** `|¿|D` `|ónde` `| est` `|á` `| la` `| bibli` `|oteca` `|?`
*   **Analysis:** Notice that `Dónde` and `biblioteca` are broken into smaller pieces. This might be because these specific forms were less common than their root words in the training data. This sentence uses 9 tokens to represent 5 words.

---

### 5. Code and Special Characters
Tokenizers are trained on code, so they are good at breaking it down into logical units.

*   **Text:** `for i in range(10):`
*   **Tokens:** `|for` `| i` `| in` `| range` `|(` `|10` `|):`
*   **Analysis:** The tokenizer correctly separates the keywords (`for`, `in`, `range`), the variable (`i`), the number (`10`), and the special characters `(` and `):`. This is crucial for the model's ability to understand and write code.

---

### 6. The "Token Cost" of a Conversation
The number of tokens directly impacts how much you pay for an API and how much of a conversation the model can "remember" (its context window).

*   **Consider this short conversation:**
    *   **User:** "Who was the first man on the moon?" (8 tokens)
    *   **Assistant:** "The first man on the moon was Neil Armstrong." (9 tokens)
    *   **User:** "What was the name of his spacecraft?" (7 tokens)

*   **To ask the second question, your application must send the *entire history* to the stateless API:**
    *   "Who was the first man on the moon? The first man on the moon was Neil Armstrong. What was the name of his spacecraft?"
    *   **Total Input Tokens for the second question:** 8 + 9 + 7 = **24 tokens**.
*   **Analysis:** Even though your second question was only 7 tokens long, you had to "pay" for 24 tokens of input context to ensure the model knew what "his spacecraft" referred to. This is why managing context length is so important when building AI applications.
