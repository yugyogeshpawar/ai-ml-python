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
