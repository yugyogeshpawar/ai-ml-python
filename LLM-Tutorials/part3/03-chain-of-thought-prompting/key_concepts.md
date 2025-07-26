# Key Concepts: Chain-of-Thought Prompting

Here are the most important terms from this lesson, explained in simple English.

### 1. Chain of Thought (CoT)
-   **What it is:** A prompting technique where you instruct the LLM to explain its reasoning process step-by-step before giving a final answer.
-   **Analogy:** "Showing your work" on a math test. Instead of just writing the final number, you write down every calculation you made to get there. This makes you less likely to make a mistake and makes it easy to see where you went wrong if you do.
-   **Why it matters:** It dramatically improves an LLM's ability to solve complex problems that require logic, math, or multiple steps of reasoning. It forces the model to slow down and "think" more deliberately.

### 2. Zero-Shot CoT
-   **What it is:** The simplest way to trigger a chain of thought, by adding a simple phrase to your prompt without providing any examples.
-   **The Magic Phrase:** The most common and effective instruction is simply adding **"Let's think step by step."** at the end of your question.
-   **Analogy:** Telling a student, "Before you give me the answer, walk me through how you're going to solve it." You're not showing them how, just prompting them to explain their own process.
-   **When to use it:** This is a great first-line approach for any problem that seems complex. It's easy to add and often provides a significant boost in accuracy.

### 3. Few-Shot CoT
-   **What it is:** A more advanced technique where you provide the LLM with one or more examples that include not just the question and answer, but also the step-by-step reasoning.
-   **Analogy:** Giving a student a workbook that has a few problems already solved with the full "show your work" process detailed. The student learns the *pattern* of how to reason about that type of problem from the examples.
-   **When to use it:** For very complex, novel, or multi-step problems where the reasoning path isn't obvious. This gives the model a clear template for how to structure its own chain of thought.
