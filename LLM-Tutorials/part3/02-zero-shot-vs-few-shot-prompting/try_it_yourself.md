# Try It Yourself: The Power of Examples

This exercise will let you experience the dramatic difference between zero-shot and few-shot prompting for a creative and slightly ambiguous task.

---

### The Scenario

Your goal is to get an LLM to generate a company slogan based on a short description. This is a creative task where examples can provide powerful guidance.

### Exercise: From Zero to Few-Shot Slogans

You will do this in two steps using your favorite AI chatbot.

**Step 1: The Zero-Shot Attempt**

*   **Go to your chatbot and give it a simple, zero-shot prompt:**
    > **Task:** Write a company slogan.
    >
    > **Company:** "Quick-Step"
    > **Description:** We make comfortable shoes for busy city walkers.
    >
    > **Slogan:**

*   **Analyze the output:** The AI will likely generate a decent but possibly generic slogan. It might be something like:
    *   "Quick-Step: Comfort in Every Step."
    *   "Walk the City with Quick-Step."
    *   "Your Busy Feet Deserve Quick-Step."
    
    These are okay, but they might not have the specific style or "punch" you're looking for.

---

**Step 2: The Few-Shot Masterstroke**

*   **Start a new chat** to ensure the context is fresh.
*   **Now, let's craft a few-shot prompt.** We will give the AI three examples of the *style* of slogan we want. Our examples will be short, catchy, and action-oriented.

    > **Task:** Write a company slogan based on the company name and description.
    >
    > ---
    >
    > **Company:** "Sun-Brew"
    > **Description:** A coffee that is bright and optimistic.
    > **Slogan:** Start your day bright.
    >
    > ---
    >
    > **Company:** "Ever-Green"
    > **Description:** Sustainable, long-lasting home goods.
    > **Slogan:** Buy it once.
    >
    > ---
    >
    > **Company:** "Apex"
    > **Description:** Climbing gear for serious athletes.
    > **Slogan:** Reach your peak.
    >
    > ---
    >
    > **Company:** "Quick-Step"
    > **Description:** We make comfortable shoes for busy city walkers.
    > **Slogan:**

*   **Analyze the output:** The AI's response should now be dramatically different. It has learned the *pattern* from your examples. It understands you want a short, punchy, verb-focused slogan that often plays on the company name. The new suggestions might be:
    *   Own the sidewalk.
    *   Outpace the city.
    *   Your city, faster.

    Notice how these feel much more specific and stylish. The AI learned the *style*, not just the task, from your examples.

**Reflection:**
This exercise demonstrates the core power of few-shot prompting. You are not just telling the AI what to do; you are showing it *how* to do it. This technique of "in-context learning" is essential for guiding the model toward creative, nuanced, or highly specific outputs that a zero-shot prompt could never reliably produce.
