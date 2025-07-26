# Try It Yourself: Building Better Prompts

This exercise is about putting the R-T-C-F (Role, Task, Context, Format) framework into practice. You will take a simple, vague prompt and progressively improve it.

---

### The Scenario

Your goal is to get help from an AI to create a healthy meal plan.

### Exercise: From Vague to Valuable

You will do this in four steps using your favorite AI chatbot.

**Step 1: The Vague Prompt**

*   **Go to your chatbot and type in the most basic prompt you can think of:**
    > "Give me a healthy meal plan."

*   **Analyze the output:** What did you get? It's probably a very generic, one-size-fits-all plan. It might include foods you don't like or can't eat. It's not very useful because you gave the AI almost nothing to work with.

---

**Step 2: Adding Role and Task**

*   **Start a new chat** to ensure the context is fresh.
*   **Now, let's add a clear Role and a more specific Task.**
    > **[Role]** You are a professional nutritionist.
    > **[Task]** Create a 3-day meal plan for me.

*   **Analyze the output:** This should be a bit better. By telling the AI to be a nutritionist, you've encouraged it to use more expert language and create a more balanced plan. But it's still generic.

---

**Step 3: Adding Context**

*   **Start a new chat again.**
*   **Now, let's add all the crucial Context the AI was missing.**
    > **[Role]** You are a professional nutritionist.
    > **[Task]** Create a 3-day meal plan for me.
    > **[Context]** I am a 35-year-old male trying to lose about 10 pounds. I work a desk job, so I'm not very active during the day, but I do a 30-minute workout 3 times a week. I am allergic to nuts and I dislike fish. I prefer quick and easy meals, especially for breakfast and lunch.

*   **Analyze the output:** This will be a dramatic improvement. The meal plan should now be tailored to your specific goals (weight loss), constraints (allergies, dislikes), and lifestyle (desk job, quick meals). The AI is no longer guessing.

---

**Step 4: Adding Format**

*   **Let's do one final version.** This time, we'll add a specific Format to make the output perfectly structured for our needs.
    > **[Role]** You are a professional nutritionist.
    > **[Task]** Create a 3-day meal plan for me.
    > **[Context]** I am a 35-year-old male trying to lose about 10 pounds. I work a desk job, so I'm not very active during the day, but I do a 30-minute workout 3 times a week. I am allergic to nuts and I dislike fish. I prefer quick and easy meals, especially for breakfast and lunch.
    > **[Format]** Please present the meal plan in a markdown table. The columns should be: "Day", "Breakfast", "Lunch", "Dinner", and "Snack". At the end of the table, please include a short grocery list for all the ingredients needed.

*   **Analyze the final output:** This is the gold standard. The information is not only tailored to you, but it's also presented in a clean, easy-to-read format, and it even includes a helpful grocery list. You've saved yourself a ton of time and gotten a genuinely useful result.

**Reflection:**
By going through these steps, you've experienced the difference between a vague request and a clear, detailed instruction. The R-T-C-F framework is a mental checklist you can use for any task to ensure you're providing the AI with everything it needs to give you the best possible response.
