# Key Concepts: Common Prompting Mistakes

Here are the key takeaways from this lesson to help you avoid common pitfalls.

### 1. Ambiguity
-   **What it is:** Using vague, unclear, or imprecise language in your prompt.
-   **The Mistake:** "Tell me about AI."
-   **The Fix:** Be specific. Define exactly what you want the AI to do. "Explain the concept of 'self-attention' in a Transformer model using a simple analogy."
-   **Why it matters:** LLMs are literal. They can't read your mind, so they need explicit instructions to avoid guessing what you want.

### 2. Context-Dropping
-   **What it is:** Assuming the AI remembers information from previous conversations or chats.
-   **The Mistake:** Starting a new chat and saying, "Summarize what we just talked about."
-   **The Fix:** Provide all necessary context within a single prompt. If needed, paste in relevant text from previous conversations.
-   **Why it matters:** Most chatbots have no long-term memory between sessions. Each chat is a blank slate, so you must provide all the information it needs to complete the current task.

### 3. Overly Complex Phrasing
-   **What it is:** Using convoluted sentences, jargon, or multiple negatives that can confuse the AI.
-   **The Mistake:** "Fail to exclude non-essential information."
-   **The Fix:** Use simple, direct, and positive phrasing. "Only include essential information."
-   **Why it matters:** Clear and simple instructions lead to clear and simple outputs. Complexity can lead to misinterpretation.

### 4. Hallucination
-   **What it is:** The tendency of an LLM to generate text that sounds plausible but is factually incorrect or nonsensical.
-   **The Mistake:** Asking the AI for a historical date or a scientific fact and trusting the answer completely without checking it.
-   **The Fix:** **Always verify.** Use the LLM as a starting point or a brainstorming partner, but cross-reference any critical information with reliable, primary sources.
-   **Why it matters:** LLMs are optimized to be plausible, not truthful. This is the single most important limitation to remember when using them for factual research.
