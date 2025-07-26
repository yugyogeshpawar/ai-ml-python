# Part 3: The Art of Talking to AI: Prompt Engineering
## Topic 4: Common Prompting Mistakes (And How to Fix Them)

You've learned the building blocks of a great prompt and some powerful techniques to guide the AI. Now, let's look at the other side of the coin: the common pitfalls that lead to confusing, incorrect, or low-quality responses.

Avoiding these mistakes is just as important as using the right techniques.

---

### Mistake 1: Vague or Ambiguous Language

LLMs are very literal. They don't have human intuition to fill in the gaps of a vague request.

*   **The Mistake:** "Tell me about Napoleon."
    *   *Problem:* What about him? His childhood? His military strategies? His exile? His impact on law? The AI has to guess, so it will give you a generic, Wikipedia-style summary.

*   **The Fix: Be Specific.** Use the R-T-C-F framework to narrow down your request.
    *   **Better Prompt:** "You are a military historian **[Role]**. **Summarize** **[Task]** Napoleon's three most significant military victories **[Context]**. Please present them as a **numbered list**, with a brief explanation for each one's importance **[Format]**."

### Mistake 2: Assuming Prior Knowledge

Never assume the AI knows about the context of your specific conversation or previous chats. Each new chat session is (usually) a blank slate.

*   **The Mistake:** You have a long conversation about marketing strategies. You start a new chat and ask, "Summarize our discussion."
    *   *Problem:* The AI has no memory of the previous chat. It will tell you it can't help because there was no discussion.

*   **The Fix: Provide All Necessary Context.** If you need to refer to something, paste it directly into the prompt.
    *   **Better Prompt:** "I'm going to paste a conversation I had about marketing. Please summarize the key takeaways. [Paste the full text of the previous conversation here]."

### Mistake 3: Using Complex or Convoluted Phrasing

While you should be specific, you should also be clear. Using overly complex sentences, jargon, or double negatives can confuse the model.

*   **The Mistake:** "Could you not refrain from failing to include any examples that aren't related to marketing?"
    *   *Problem:* This sentence is a mess of negatives. The AI might misinterpret it and do the exact opposite of what you want.

*   **The Fix: Use Simple, Direct Language.**
    *   **Better Prompt:** "Please provide only marketing-related examples."

### Mistake 4: Asking for Opinions or Feelings

LLMs do not have personal beliefs, opinions, or feelings. They are pattern-matching machines. When you ask for an opinion, they are simply generating text that mimics how a human might express an opinion based on their training data.

*   **The Mistake:** "Do you think that remote work is a good idea?"
    *   *Problem:* The AI doesn't "think." It will give you a balanced summary of common arguments for and against remote work, but it's not a personal belief.

*   **The Fix: Ask for Analysis or Perspectives.** Frame the question in a more objective way.
    *   **Better Prompt:** "Analyze the primary arguments for and against remote work, citing common points made by both proponents and critics."

### Mistake 5: Accepting the First Answer as Fact

This is the most critical mistake to avoid. **LLMs can and do make things up.** This is often called "hallucination." Because their fundamental job is to generate plausible-sounding text, they will sometimes generate text that sounds completely reasonable but is factually incorrect.

*   **The Mistake:** You ask, "What was the score of the 1992 Super Bowl?" The AI gives you an answer, and you use it in a report without checking.
    *   *Problem:* The AI might have mixed up the year, the teams, or the score. It has no concept of "truth," only of what patterns of text are probable.

*   **The Fix: Always Verify Critical Information.** Treat the LLM as a highly knowledgeable but sometimes unreliable assistant. Use it to find leads, summarize information, and generate ideas, but always cross-reference important facts with a trusted source like a search engine, a textbook, or an academic paper.

By being aware of these common mistakes, you can refine your prompting style, get more reliable results, and use AI tools more effectively and responsibly.
