# Try It Yourself: Forcing the AI to "Show Its Work"

This exercise will clearly demonstrate the power of Chain-of-Thought prompting on a simple logic puzzle that LLMs often fail at without it.

---

### The Scenario

We're going to give the AI a classic logic puzzle. These are often tricky for LLMs because they require careful tracking of relationships and constraints, and they can't be solved by just retrieving a fact from the training data.

**The Puzzle:**
> John has a brother named Mark and a sister named Sarah. Sarah has two daughters, Lucy and Mia. Mark has one son, Tom. What is the relationship between John and Tom?

### Exercise: From Failure to Success with CoT

You will do this in two steps with your favorite AI chatbot.

**Step 1: The Standard (Zero-Shot) Prompt**

*   **Go to your chatbot and give it the puzzle directly.**
    > John has a brother named Mark and a sister named Sarah. Sarah has two daughters, Lucy and Mia. Mark has one son, Tom. What is the relationship between John and Tom?

*   **Analyze the output.** Pay close attention to the answer. Many powerful LLMs will get this wrong! They might say:
    *   "John is Tom's brother." (Incorrect)
    *   "John is Tom's cousin." (Incorrect)
    *   "There is no relationship mentioned." (Incorrect)
    *   Some might get it right, but it's often unreliable.

*   **Why does it fail?** The model is trying to jump straight to the answer. It sees the words "John," "brother," and "Tom" and makes a quick, surface-level association. It doesn't take the time to trace the family tree properly.

---

**Step 2: The Chain-of-Thought Prompt**

*   **Start a new chat** to make sure the context is clean.
*   **Now, ask the exact same question, but add the magic phrase at the end.**
    > John has a brother named Mark and a sister named Sarah. Sarah has two daughters, Lucy and Mia. Mark has one son, Tom. What is the relationship between John and Tom?
    >
    > **Let's think step by step.**

*   **Analyze the output.** The difference should be immediate and obvious. The model will now "show its work" before giving the final answer. The output will look something like this:

    > 1.  John has a brother named Mark. This means John and Mark are siblings.
    > 2.  Mark has a son named Tom. This means Mark is Tom's father.
    > 3.  Since John is Mark's brother, and Mark is Tom's father, John is Tom's uncle.
    >
    > **Therefore, John is Tom's uncle.**

*   **Why does it succeed?** By forcing the model to generate the intermediate reasoning steps, we've created a logical "chain" for it to follow. Each step is a simple, factual statement derived from the previous one. This makes it much easier for the model to arrive at the correct final conclusion. It's no longer guessing; it's reasoning.

**Reflection:**
This simple exercise proves that Chain-of-Thought is not just a neat trick; it's a fundamental technique for improving the reliability of LLMs on any task that requires logical deduction. Before you trust an AI's answer to a complex question, ask it to show you how it got there.
