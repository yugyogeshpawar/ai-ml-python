# Interview Questions: Prompt Templates

### Q1: What is a Prompt Template and why is it a crucial component in LangChain?

**Answer:**

A Prompt Template is a reusable and dynamic template for generating prompts for language models. It's a string that contains instructions, examples, and variables that can be filled in at runtime.

They are crucial for several reasons:

*   **Reusability:** You can define a prompt structure once and reuse it with different inputs, which keeps your code DRY (Don't Repeat Yourself).
*   **Clarity and Maintenance:** They separate the prompt's logic from the application code, making both easier to read and maintain. If you need to tweak the prompt, you only have to change it in one place.
*   **Dynamic Prompts:** They allow you to create prompts that are tailored to specific user inputs or contexts, which is essential for building interactive and responsive applications.
*   **Consistency:** They ensure that the prompts sent to the LLM follow a consistent structure, which can lead to more predictable and reliable outputs.

### Q2: How do you define a Prompt Template with multiple input variables?

**Answer:**

You can define a `PromptTemplate` with multiple input variables by including all the variable names in the `input_variables` list and referencing them in the `template` string.

Here's an example:

```python
from langchain import PromptTemplate

multi_variable_template = PromptTemplate(
    input_variables=["company", "product"],
    template="What is a good slogan for a company named '{company}' that sells '{product}'?"
)

formatted_prompt = multi_variable_template.format(
    company="SoleMates",
    product="colorful socks"
)

print(formatted_prompt)
# Expected output: "What is a good slogan for a company named 'SoleMates' that sells 'colorful socks'?"
```

This allows you to create complex and highly contextual prompts by combining multiple pieces of information.

### Q3: What are "few-shot" prompts, and how can Prompt Templates be used to create them?

**Answer:**

"Few-shot" prompts are a technique used to improve the performance of LLMs by providing them with a few examples of the desired input/output format within the prompt itself. This helps the model understand the task and produce more accurate and consistently formatted responses.

`FewShotPromptTemplate` in LangChain is specifically designed for this purpose. You provide it with:

*   A set of examples (the "shots").
*   An `ExampleSelector` to choose which examples to include in the prompt (optional).
*   A `PromptTemplate` to format each example.
*   A prefix and a suffix to wrap around the formatted examples.

Here's a simplified conceptual example:

```python
from langchain import FewShotPromptTemplate, PromptTemplate

# 1. Create a list of examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

# 2. Create a template to format each example
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# 3. Create the FewShotPromptTemplate
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of every input.",
    suffix="Input: {user_input}\nOutput:",
    input_variables=["user_input"]
)

# 4. Format the prompt with user input
final_prompt = few_shot_template.format(user_input="hot")

print(final_prompt)
```

**Expected Output:**

```
Give the antonym of every input.
Input: happy
Output: sad
Input: tall
Output: short
Input: hot
Output:
```

This prompt now clearly demonstrates the task to the LLM, making it much more likely to respond with "cold".
