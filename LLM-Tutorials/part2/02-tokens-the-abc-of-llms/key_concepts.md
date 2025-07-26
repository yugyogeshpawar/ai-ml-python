# Key Concepts: Tokens

Here are the most important terms from this lesson, explained in simple English.

### 1. Token
-   **What it is:** A piece of text that an LLM treats as a single unit. It can be a whole word, a part of a word (like `pre` or `ing`), or a punctuation mark.
-   **Analogy:** The LEGO bricks of language. Instead of having a unique brick for every object in the world, you have a set of standard bricks that can be combined to build anything you can imagine. Tokens are the standard bricks that LLMs use to build and understand any word or sentence.
-   **Why it matters:** Tokens are how LLMs "see" text. The entire process of language understanding and generation is based on processing sequences of these tokens.

### 2. Tokenizer
-   **What it is:** A tool that breaks a piece of text down into a sequence of tokens. It also does the reverse, converting a sequence of tokens back into human-readable text.
-   **Analogy:** A translator at the border of "Human-Land" and "AI-Land." When you give the AI a sentence, the tokenizer translates it into the language of tokens that the AI can understand. When the AI replies, the tokenizer translates the AI's token-language back into a human sentence.
-   **Why it matters:** It's the essential first step in any interaction with an LLM. The way a sentence is "tokenized" can sometimes affect how the model responds.

### 3. Context Window
-   **What it is:** The maximum number of tokens that an LLM can "remember" or process at one time. This includes both your input (prompt) and the model's output (response).
-   **Analogy:** The short-term memory of the AI. If you have a conversation that exceeds its context window, the AI will start to "forget" the beginning of the conversation, just like a person might forget the start of a very long story.
-   **Why it matters:** It defines the limits of the model's capabilities for a single task. A model with a large context window can read and analyze entire books, while a model with a small context window can only handle a few paragraphs at a time. It's also a key factor in how much it costs to use the model.
