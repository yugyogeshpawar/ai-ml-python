# Exercises: Chains in LangChain

These exercises will help you understand how to build and run chains.

### Exercise 1: Build a Simple `LLMChain`

1.  Create a new Python script.
2.  Your goal is to create a simple "historian" bot.
3.  Create a `PromptTemplate` that takes a single variable, `era`, and asks for a brief summary of that historical period. (e.g., "Briefly describe the key events of the {era} era.").
4.  Initialize an `OpenAI` LLM.
5.  Create an `LLMChain` that combines your prompt template and the LLM.
6.  Run the chain with the input "Renaissance" and print the result.

### Exercise 2: Create a `SimpleSequentialChain`

Let's build a two-step chain that first generates a company name and then creates a slogan for it.

1.  **Chain 1: Name Generator**
    *   Create an `LLMChain` that takes a `product` as input and generates a `company_name`.
    *   The prompt should be something like: "What is a good name for a company that makes {product}?"

2.  **Chain 2: Slogan Generator**
    *   Create a second `LLMChain` that takes the `company_name` as input and generates a slogan.
    *   The prompt should be: "Write a catchy slogan for a company named {company_name}."

3.  **Combine them with `SimpleSequentialChain`**
    *   Import `SimpleSequentialChain` from `langchain.chains`.
    *   Create an instance of `SimpleSequentialChain`, passing in your two chains in the correct order.
    *   Run the sequential chain with a product like "organic dog food" and print the final output (which should be the slogan).

### Exercise 3: Debug a Chain

Imagine you have the following code, but it's not working.

```python
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI()

# Note the typo in the input_variables list
prompt = PromptTemplate(
    input_variables=["conpany"], # Typo here!
    template="What does the company {company} do?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# This will raise an error
print(chain.run("Apple"))
```

1.  Run this code and observe the error message. What does it tell you?
2.  Identify the bug in the `PromptTemplate` definition.
3.  Fix the bug and run the code again to ensure it works correctly.

This exercise highlights the importance of ensuring the `input_variables` in your `PromptTemplate` match the variables used in the `template` string.
