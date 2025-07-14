# Project: Email Summarizer

This project demonstrates how to build a LangChain chain that takes the content of an email and generates a concise summary.

## Objective

The goal is to create a chain that can:
1.  Take a long email text as input.
2.  Use a prompt template to instruct the LLM to summarize the email.
3.  Use an output parser to structure the summary into key points.
4.  Provide a clean, easy-to-read summary of the email.

## Step-by-Step Implementation

### 1. Create the Main Script

Now, let's create the main Python script for our application.

**`summarizer.py`**

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. Create a sample email
email_content = """
Subject: Project Phoenix - Weekly Update

Hi Team,

This week, we've made significant progress on the new user authentication module. The front-end components are 80% complete, and the back-end API is fully functional. We encountered a minor setback with the database migration, which has caused a 2-day delay. We expect to resolve this by Wednesday.

Next week, we will focus on integrating the front-end and back-end, and we'll begin user acceptance testing (UAT). Please ensure all your code is committed by EOD Friday.

Action Items:
- Alice: Finalize the UI for the password reset page.
- Bob: Deploy the latest API version to the staging environment.
- Charlie: Prepare the UAT test cases.

Best,
Project Manager
"""

# 2. Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "Summarize the following email into 3-5 key bullet points:\n\n{email}"
)

# 3. Initialize the Chat Model
model = ChatOpenAI(temperature=0)

# 4. Create an output parser
output_parser = StrOutputParser()

# 5. Build the LCEL chain
chain = prompt | model | output_parser

# 6. Run the chain
summary = chain.invoke({"email": email_content})

# 7. Print the result
print("--- Original Email ---")
print(email_content)
print("\n--- Generated Summary ---")
print(summary)
```

### 2. How to Run the Project

1.  Make sure you have your `OPENAI_API_KEY` environment variable set.
2.  Navigate to this directory in your terminal.
3.  Run the script:

    ```bash
    python summarizer.py
    ```

4.  The script will print the original email and the generated summary.

This project demonstrates a practical application of LangChain for text summarization, a common and valuable use case.
