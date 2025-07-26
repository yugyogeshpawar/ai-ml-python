# Part 3: The Art of Talking to AI: Prompt Engineering
## Topic 3: Chain-of-Thought Prompting

We've seen how to give the AI a role, a task, context, and examples. But what about complex problems that require multiple steps of reasoning? If you ask an LLM a multi-step math problem directly, it will often get the answer wrong.

Why? Because it's trying to predict the final answer in one go, and that's not how reasoning works. Humans don't instantly know the answer to `(15 * 3) + (10 / 2) - 7`. We work it out step-by-step.

**Chain-of-Thought (CoT) prompting** is a simple but incredibly powerful technique that encourages the LLM to do the same: to "think step-by-step" before giving the final answer.

---

### The "Let's think step by step" Trick

The amazing discovery behind CoT prompting is that by simply adding the phrase **"Let's think step by step"** (or a similar instruction) to your prompt, you can dramatically improve the model's performance on tasks that require logic, math, or complex reasoning.

> **Simple Definition:** Chain-of-Thought prompting is the practice of instructing the model to break down a complex problem into a series of intermediate reasoning steps, rather than trying to answer it all at once.

**Analogy: Showing Your Work in a Math Class**

Remember in school when your math teacher would insist that you "show your work"? You couldn't just write down the final answer, even if you thought you knew it. You had to write out each step of your calculation.

Why did they make you do this?
1.  **It slowed you down.** It forced you to be more deliberate and less likely to make a careless mistake.
2.  **It made your reasoning visible.** If you got the answer wrong, you (and the teacher) could look at your steps and see exactly where you made the error.
3.  **It reinforced the process.** It trained you to follow a logical sequence to solve problems.

Chain-of-Thought prompting does the exact same thing for an LLM. It forces the model to generate a sequence of reasoning tokens that lead to the final answer, which makes it far more likely that the final answer will be correct.

---

### How to Use Chain-of-Thought Prompting

There are two main ways to use CoT: zero-shot and few-shot.

#### 1. Zero-Shot CoT

This is the simplest method and often works surprisingly well. You just add a simple instruction to your prompt.

**Standard Prompt (Often Fails):**
> **Question:** A juggler has 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls does the juggler have?
>
> **Answer:**

An LLM might rush to an answer and get it wrong.

**Zero-Shot CoT Prompt (Much Better):**
> **Question:** A juggler has 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls does the juggler have?
>
> **Let's think step by step.**
>
> **Answer:**

By adding that one simple sentence, you are instructing the model to output its reasoning process first. It will likely generate something like:

> 1.  The juggler starts with 16 balls in total.
> 2.  Half of the balls are golf balls, so we calculate 16 / 2 = 8. There are 8 golf balls.
> 3.  Half of the golf balls are blue, so we calculate 8 / 2 = 4. There are 4 blue golf balls.
>
> **The final answer is 4.**

The model has externalized its "thought process," making it much more accurate.

#### 2. Few-Shot CoT

For even more complex or novel problems, you can combine Chain-of-Thought with few-shot prompting. You provide examples that not only show the correct answer but also demonstrate the step-by-step reasoning process.

**Few-Shot CoT Prompt:**
> **Question:** [Example Question 1]
> **Answer:** [Step-by-step reasoning for Q1]. The final answer is [Answer 1].
>
> ---
>
> **Question:** [Example Question 2]
> **Answer:** [Step-by-step reasoning for Q2]. The final answer is [Answer 2].
>
> ---
>
> **Question:** [Your new, complex question]
> **Answer:**

This is the most robust prompting method for complex reasoning tasks. You are not only telling the model to think step-by-step, but you are also showing it *exactly* what that thinking process should look like.

In the next lesson, we'll look at some of the common mistakes people make when prompting and how to avoid them.
