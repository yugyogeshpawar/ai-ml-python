# Examples in Action: Your First AI App

The "Poem Generator" from the main lesson is a great start. Here are a few other simple but powerful applications you could build in Google Colab using the same basic pattern.

**The Core Pattern:**
1.  Import the `openai` library.
2.  Set up your API key.
3.  Define a clear prompt.
4.  Make the API call using `client.chat.completions.create()`.
5.  Extract and print the response.

---

### Example 1: The "Explain Like I'm 5" App

*   **Goal:** An app that can explain any complex topic in very simple terms.
*   **Code Modification:**
    *   Get two inputs from the user: the topic and the target age.
        ```python
        user_topic = input("What topic do you want to understand? ")
        user_age = input("What age should I explain it for (e.g., 5, 10, 15)? ")
        ```
    *   Change the prompt to be more dynamic.
        ```python
        prompt = f"Explain the concept of '{user_topic}' to me as if I were {user_age} years old."
        ```
*   **Use Case:** This is a fantastic learning tool. You can use it to understand anything from "black holes" to "the stock market" at the right level for you.

---

### Example 2: The "Email Polisher" App

*   **Goal:** An app that takes a rough draft of an email and makes it sound more professional.
*   **Code Modification:**
    *   Get the user's rough draft as input. It's better to ask them to paste it in after a prompt rather than using `input()` for multi-line text.
        ```python
        print("Please paste your rough email draft below, and press Ctrl+D (or Ctrl+Z on Windows) when you are done.")
        import sys
        rough_draft = sys.stdin.read()
        ```
    *   Create a detailed prompt that gives the AI a clear role and task.
        ```python
        prompt = f"""
        You are an expert business communication coach. 
        Please rewrite the following email draft to make it more professional, clear, and concise. 
        Keep the core message the same.

        Rough Draft:
        "{rough_draft}"
        """
        ```
*   **Use Case:** This helps you quickly turn bullet points or casual thoughts into well-written, professional emails, saving time and improving your communication.

---

### Example 3: The "Recipe from Ingredients" App

*   **Goal:** An app that suggests a recipe based on ingredients you have in your fridge.
*   **Code Modification:**
    *   Get a list of ingredients from the user.
        ```python
        user_ingredients = input("List the ingredients you have, separated by commas: ")
        ```
    *   Use a prompt that asks for a recipe and provides the ingredients as context.
        ```python
        prompt = f"""
        You are a creative chef. 
        Create a simple recipe using only the following ingredients. 
        Please provide a name for the dish, a list of the ingredients, and step-by-step instructions.

        Available Ingredients: {user_ingredients}
        """
*   **Use Case:** This is a practical tool to help reduce food waste and solve the daily "what should I make for dinner?" problem.
