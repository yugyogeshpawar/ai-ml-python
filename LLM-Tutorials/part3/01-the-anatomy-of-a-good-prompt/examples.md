# Examples in Action: The Anatomy of a Good Prompt

Let's break down the R-T-C-F (Role, Task, Context, Format) framework with a few more practical, before-and-after examples.

### Example 1: Brainstorming Ideas

*   **Goal:** Get ideas for a child's birthday party.
*   **Bad Prompt (Vague):**
    > "Ideas for a birthday party."

*   **Good Prompt (Using R-T-C-F):**
    > **[Role]** You are an expert party planner for children's events.
    > **[Task]** Brainstorm 5 creative and unique theme ideas for a birthday party.
    > **[Context]** The party is for my son who is turning 7. He loves science, space, and building things with LEGOs. We have a budget of $200 for about 10 kids and the party will be at our home with a backyard.
    > **[Format]** Please provide the answer as a numbered list. For each theme, include a catchy name, a brief description, and one suggestion for a simple, DIY activity.

*   **Analysis of the Improvement:** The bad prompt would likely give generic ideas like "superhero party" or "pirate party." The good prompt provides all the necessary constraints (age, interests, budget, location) and specifies the exact output structure, leading to tailored and actionable ideas like a "Mad Scientist Lab" or "LEGO Space Mission" party.

---

### Example 2: Summarizing a Document

*   **Goal:** Understand the key points of a long article.
*   **Bad Prompt (Vague Task):**
    > "Tell me about this article: [paste article text]"

*   **Good Prompt (Using R-T-C-F):**
    > **[Role]** You are a research assistant skilled at synthesizing complex information into clear summaries.
    > **[Task]** Summarize the following article. Your summary should identify the main argument, the key pieces of evidence used, and the author's conclusion.
    > **[Context]** The article is: [paste article text]. My goal is to quickly understand if this article is relevant for my own research paper on climate change policy.
    > **[Format]** Please structure your response into three distinct sections with the following headers: "Main Argument," "Key Evidence," and "Conclusion."

*   **Analysis of the Improvement:** The bad prompt is ambiguous. "Tell me about" could mean anything. The good prompt defines the exact *kind* of summary needed, making the output far more useful for the user's specific goal. Adding the user's goal as context helps the AI prioritize the most relevant information.

---

### Example 3: Writing Code

*   **Goal:** Get a Python script to help with a task.
*   **Bad Prompt (Missing Context and Format):**
    > "Write a Python script that renames files."

*   **Good Prompt (Using R-T-C-F):**
    > **[Role]** You are an expert Python developer who writes clean, well-commented code.
    > **[Task]** Write a Python script that renames all the `.jpg` files in a specific folder.
    > **[Context]** The script should rename the files from their current names (e.g., `IMG_1234.jpg`) to a new format: `Vacation-2024-001.jpg`, `Vacation-2024-002.jpg`, and so on. The user should be prompted to enter the folder path when they run the script.
    > **[Format]** Please provide only the Python code in a single code block. Add comments to the code to explain what each part does.

*   **Analysis of the Improvement:** The bad prompt is dangerous; a script that "renames files" without specific instructions could cause chaos on a user's computer. The good prompt provides exact specifications for the input and output file names, specifies the target file type, and asks for user interaction and comments, resulting in a safe, useful, and understandable piece of code.
