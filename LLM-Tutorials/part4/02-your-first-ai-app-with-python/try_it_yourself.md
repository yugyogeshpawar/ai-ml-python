# Try It Yourself: Modifying Your First AI App

This exercise is all about experimenting with the code you've just written. By making small changes, you can see how much control you have over the AI's output and the application's behavior.

---

### How to Do This Exercise

Go back to your Google Colab notebook with the "Poem Generator" code. For each modification below, change the code in the cell and run it again to see what happens.

---

### Modification 1: Change the AI's Persona

The current "system" prompt tells the AI it's a "helpful poet." Let's change its personality.

*   **Find this line:**
    ```python
    {"role": "system", "content": "You are a helpful poet."},
    ```
*   **Change the content to something different.** Try one of these:
    *   `"You are a grumpy, old-fashioned poet who hates writing poems."`
    *   `"You are a pirate who speaks only in pirate slang."`
    *   `"You are a Shakespearean playwright."`
*   **Run the code again.** Ask for a poem about "the ocean." How does the tone and style of the poem change based on the new persona?

---

### Modification 2: Change the Prompt's Task

Right now, the prompt asks for a "short, four-line poem." Let's ask for something else.

*   **Find this line:**
    ```python
    prompt = f"Write a short, four-line poem about {user_topic}."
    ```
*   **Change the task.** Try one of these ideas:
    *   `prompt = f"Write a haiku (5-7-5 syllables) about {user_topic}."`
    *   `prompt = f"Write a two-sentence horror story about {user_topic}."`
    *   `prompt = f"Explain the concept of {user_topic} as if I were a five-year-old."`
*   **Run the code.** See how the AI adapts to the new, more specific task.

---

### Modification 3: Use a More Powerful Model

We used `gpt-3.5-turbo`, which is fast and efficient. But what if we want higher quality? Let's switch to a more powerful (and slightly slower/more expensive) model like `gpt-4o`.

*   **Find this line:**
    ```python
    model="gpt-3.5-turbo",
    ```
*   **Change it to:**
    ```python
    model="gpt-4o",
    ```
*   **Run the code again.** Ask for a poem on the same topic you used before. Is the result more creative, nuanced, or interesting? For simple tasks, the difference might be small, but for complex requests, a more powerful model can make a huge difference.

---

### Modification 4 (Advanced): Add More User Input

Let's make our app more interactive by asking the user for more than just the topic.

*   **Add a new `input()` line** right after the first one:
    ```python
    user_topic = input("What topic should the poem be about? ")
    user_style = input("What style should the poem be in (e.g., happy, sad, funny)? ")
    ```
*   **Now, modify the f-string** to include this new variable:
    ```python
    prompt = f"Write a short, four-line {user_style} poem about {user_topic}."
    ```
*   **Run the code.** Your application is now more flexible! It can generate a `sad` poem about `cats` or a `funny` poem about `computers`.

**Reflection:**
This exercise shows that the Python code acts as a **wrapper** or a **harness** for the real magic, which is the **prompt**. By changing the variables that go into the prompt (`role`, `task`, `model`), you can get a huge variety of behaviors from the same basic code structure. This is a fundamental concept in building AI applications.
