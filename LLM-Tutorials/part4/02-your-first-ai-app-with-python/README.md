# Part 4: Building with AI: Your First Projects
## Topic 2: Your First AI App with Python

This is an exciting moment. You're about to write your first real program that interacts with a Large Language Model. We will use the tools you've learned about to create a simple "Poem Generator" application.

**Our Goal:** Write a Python script that asks the user for a topic and then uses an AI to write a short poem about that topic.

**Our Tools:**
*   **Programming Language:** Python (the most popular language for AI)
*   **Coding Environment:** Google Colab (no installation needed!)
*   **AI Model:** OpenAI's GPT model (accessed via their API)

**Prerequisites:**
*   A Google Account (for Google Colab).
*   An OpenAI API Key.
    *   You'll need to sign up for an account at [https://platform.openai.com/](https://platform.openai.com/).
    *   Navigate to the "API Keys" section in your account settings.
    *   Create a new secret key. **Important:** Copy this key and save it somewhere safe. You will not be able to see it again. (Note: Using the OpenAI API may incur small costs, but they provide a free credit for new users that is more than enough for this tutorial).

---

### Step 1: Setting Up Your Google Colab Notebook

1.  Go to [https://colab.research.google.com/](https://colab.research.google.com/).
2.  Click on "New notebook."
3.  You now have a blank, interactive Python environment. The main area is divided into "cells" where you can write and run code.

---

### Step 2: Installing the OpenAI Library

The first thing we need to do in any Python project is install the necessary libraries. In our case, we just need the official OpenAI Python SDK.

*   **In the first code cell, type the following command and run it** (by clicking the play button or pressing Shift+Enter):

```python
!pip install openai
```

*   **What this does:** `pip` is Python's package manager. The `!` at the beginning tells Colab that this is a shell command, not Python code. This command downloads the `openai` library from the internet and installs it into your Colab environment so you can use it in your script.

---

### Step 3: The Python Code

Now we'll write the actual program. We will do this in a new code cell. Below is the complete code, followed by a line-by-line explanation of what it does.

*   **Copy and paste the following code into the next code cell:**

```python
# 1. Import the necessary libraries
import os
from openai import OpenAI

# 2. Set up your API key
# It's best practice to set the key as an environment variable.
# In Colab, you can use the "Secrets" tab (key icon on the left) 
# to store your key securely. Name the secret "OPENAI_API_KEY".
# Then, run this cell.
# If you don't use secrets, you can uncomment the line below and paste your key directly.
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE" 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 3. Get input from the user
# The input() function displays a message and waits for the user to type something.
user_topic = input("What topic should the poem be about? ")

# 4. Create the prompt using an f-string
# An f-string lets you easily embed variables directly into a string.
prompt = f"Write a short, four-line poem about {user_topic}."
print(f"\nMy prompt to the AI: '{prompt}'\n")

# 5. Make the API call to the AI model
# This is where we send our request to the OpenAI "waiter".
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # You can use "gpt-4o" for a more powerful model
    messages=[
        {"role": "system", "content": "You are a helpful poet."},
        {"role": "user", "content": prompt}
    ]
)

# 6. Extract and print the AI's response
# The response from the API is a complex object. We need to navigate it to get the content.
ai_poem = response.choices[0].message.content

print("--- AI's Poem ---")
print(ai_poem)
print("-----------------")

```

---

### Line-by-Line Explanation

*   **Line 1 (`import os`, `from openai import OpenAI`):** We `import` the libraries we need. `os` is for interacting with the operating system (to get our secret API key), and `OpenAI` is the main tool from the library we just installed.

*   **Line 2 (Setting up the key):** This is the most secure way to handle your API key in Colab. You add your key as a "Secret" with the name `OPENAI_API_KEY`, and this code reads it without you having to paste your secret key directly in the code.

*   **Line 3 (`user_topic = input(...)`):** This line prints the message "What topic should the poem be about?" and pauses the program, waiting for you to type something and press Enter. Whatever you type is stored in the `user_topic` variable.

*   **Line 4 (`prompt = f"..."`):** We use an f-string to construct our final prompt. The `{user_topic}` part is automatically replaced with whatever you typed in the previous step.

*   **Line 5 (`response = client.chat.completions.create(...)`):** This is the main event! This is the API call.
    *   `model="gpt-3.5-turbo"`: We specify which AI model we want to use. GPT-3.5 Turbo is fast and cheap.
    *   `messages=[...]`: This is the prompt, formatted in the way the API expects. We provide a "system" message to set the AI's role and a "user" message containing our actual prompt.

*   **Line 6 (`ai_poem = response.choices[0].message.content`):** The `response` object from the API contains a lot of information. The actual text of the poem is buried inside it. This line navigates through the JSON structure (`choices` -> first item `[0]` -> `message` -> `content`) to extract just the text of the poem and store it in the `ai_poem` variable.

*   **Line 7 (`print(...)`):** Finally, we print the poem that the AI generated.

**Congratulations!** You have successfully built your first AI-powered application. You've used Python to get user input, send it to a powerful LLM via an API, and display the result. This simple pattern is the foundation for countless AI applications.
