# Project: Code Documentation Generator

This project demonstrates how to build a LangChain chain that takes a Python function as input and generates a professional docstring for it.

## Objective

The goal is to create a chain that can:
1.  Take the source code of a Python function as input.
2.  Use a prompt template to instruct the LLM to analyze the function and generate a docstring.
3.  The docstring should explain what the function does, its arguments, and what it returns.
4.  Use an output parser to ensure the generated docstring is well-formatted.

## Step-by-Step Implementation

### 1. Create the Main Script

Now, let's create the main Python script for our application.

**`docstring_generator.py`**

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. Create a sample Python function
function_code = """
def calculate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    else:
        list_fib = [0, 1]
        while len(list_fib) < n:
            next_fib = list_fib[-1] + list_fib[-2]
            list_fib.append(next_fib)
        return list_fib
"""

# 2. Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "Generate a professional Python docstring for the following function. The docstring should include a brief explanation of what the function does, a description of its arguments, and what it returns.\n\nFunction:\n```python\n{function}\n```"
)

# 3. Initialize the Chat Model
model = ChatOpenAI(temperature=0)

# 4. Create an output parser
output_parser = StrOutputParser()

# 5. Build the LCEL chain
chain = prompt | model | output_parser

# 6. Run the chain
docstring = chain.invoke({"function": function_code})

# 7. Print the result
print("--- Original Function ---")
print(function_code)
print("\n--- Generated Docstring ---")
print(docstring)

# You can then combine the docstring with the original function code
print("\n--- Function with Docstring ---")
print(f"def calculate_fibonacci(n):\n    \"\"\"{docstring}\"\"\"\n{function_code.split(':', 1)[1]}")
```

### 2. How to Run the Project

1.  Make sure you have your `OPENAI_API_KEY` environment variable set.
2.  Navigate to this directory in your terminal.
3.  Run the script:

    ```bash
    python docstring_generator.py
    ```

4.  The script will print the original function, the generated docstring, and the final function with the docstring included.

This project demonstrates how LangChain can be used as a powerful developer tool to automate tasks like code documentation.
