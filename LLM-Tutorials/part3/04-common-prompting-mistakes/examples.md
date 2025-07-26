# Examples in Action: Common Prompting Mistakes

Seeing examples of bad prompts and how to fix them is one of the best ways to improve your own prompting skills.

### 1. Mistake: Vagueness
Being too general or unclear about your needs.

*   **Bad Prompt:**
    > "Write a story about a dragon."

*   **Why it's bad:** What kind of story? For whom? A children's story? A dark fantasy epic? A comedy? The AI has to guess, and its guess will be generic.

*   **Good Prompt:**
    > "Write a short, heartwarming children's story (about 300 words) about a small, shy dragon who is afraid of flying. The story should have a happy ending where the dragon overcomes its fear with the help of a wise old owl friend. The tone should be gentle and encouraging."

---

### 2. Mistake: Missing Context
Assuming the AI knows something it doesn't.

*   **Bad Prompt:**
    > "Write a follow-up email to Jennifer."

*   **Why it's bad:** The AI has no idea who Jennifer is, what you discussed previously, or what the purpose of the follow-up is.

*   **Good Prompt:**
    > "I need to write a professional follow-up email to Jennifer Smith (j.smith@examplecorp.com).
    > **Context:** We had a meeting yesterday where we discussed the Q3 marketing budget. I promised to send her the final budget spreadsheet by the end of today.
    > **Task:** Write a brief, friendly email attaching the spreadsheet. Remind her that the key deadline for feedback is this Friday."

---

### 3. Mistake: Asking for Opinions
Asking the AI for subjective beliefs or feelings, which it doesn't have.

*   **Bad Prompt:**
    > "Do you think Picasso was a better artist than Monet? Why?"

*   **Why it's bad:** The AI doesn't "think" or "prefer" one artist. It will generate a generic response that just summarizes the common arguments about both artists without taking a real stance.

*   **Good Prompt:**
    > "Compare and contrast the artistic styles of Picasso and Monet. Create a two-column markdown table. The first column should list key artistic elements (e.g., Brushwork, Subject Matter, Use of Color, Key Movement). The second column should describe how each artist approached that element."

---

### 4. Mistake: Accepting a Hallucination as Fact
Trusting the AI's output without verification, especially for factual information.

*   **Bad Prompt:**
    > "What was the third movie directed by Stanley Kubrick?"

*   **AI's (Potentially Incorrect) Response:**
    > "The third movie directed by Stanley Kubrick was *Dr. Strangelove*."

*   **Why it's bad:** This is a plausible but potentially incorrect "fact." An LLM can easily get dates, numbers, and orders wrong. (Kubrick's third feature film was actually *The Killing*).

*   **Good Use of AI (for fact-finding):**
    > "Generate a list of Stanley Kubrick's feature films in chronological order."

*   **The Human Step (Crucial):** After getting this list, you would then use a reliable source (like Wikipedia or IMDb) to quickly verify the order and find the third film yourself. You use the AI to gather and structure the information, but you perform the final verification step. This combines the speed of AI with the reliability of human critical thinking.
