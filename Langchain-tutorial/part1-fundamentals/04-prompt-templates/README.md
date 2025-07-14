# 4. Prompt Templates: The Art of Crafting Reusable Prompts

In the last lesson, we sent a simple, hard-coded string to our LLM. This is fine for a single query, but it's not practical for real applications where prompts need to change based on user input or other factors. This is where **Prompt Templates** become essential.

## What are Prompt Templates?

A Prompt Template is a blueprint for a prompt. It's a piece of text that contains variables that can be filled in dynamically. Think of it as a "mail merge" or an "f-string" for your LLM prompts.

By using templates, you separate the logic of your application from the specific wording of the prompt. This makes your code cleaner, your prompts easier to manage, and your application more flexible.

A good prompt template can include:
*   **Instructions:** Telling the LLM what you want it to do.
*   **Context:** Providing relevant information or data.
*   **Few-shot examples:** Giving the model examples of the desired input/output format to improve its accuracy.
*   **Input Variables:** Placeholders for dynamic content.

## Step-by-Step Code Tutorial

The `prompt_template_example.py` script demonstrates a basic but powerful use case. Let's break it down.

### 1. Import `PromptTemplate`

```python
from langchain import PromptTemplate
```
This imports the core class we need to create our template.

### 2. Define the Template

This is the most important step. We create an instance of `PromptTemplate` with two key arguments:

```python
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```
*   **`template`**: This is the string that defines the structure of our prompt. The `{product}` part is a placeholder for a variable.
*   **`input_variables`**: This is a list of the names of all the variables that appear in the `template` string. LangChain uses this to validate that you provide all the necessary inputs later on. If you had a template like `"A {adjective} {noun}"`, your `input_variables` would be `["adjective", "noun"]`.

### 3. Format the Prompt

Now that we have our blueprint, we can create a specific prompt by "formatting" it with concrete values.

```python
formatted_prompt = prompt_template.format(product="eco-friendly water bottles")
```
The `.format()` method takes keyword arguments where the keys match the `input_variables` we defined. It returns a complete, ready-to-use prompt string:
`"What is a good name for a company that makes eco-friendly water bottles?"`

### 4. Use the Prompt

This final, formatted string is what you pass to the LLM's `.predict()` method, just as we did in the previous lesson.

## Why is This Better Than Just Using f-strings?

While you could achieve a similar result with a simple Python f-string (`f"What is a good name for a company that makes {product}?"`), `PromptTemplate` offers significant advantages in a real application:

*   **Validation:** It forces you to be explicit about your input variables, catching errors early if you forget to provide one.
*   **Composition:** Prompt templates are designed to be combined with other LangChain components, like output parsers and other chains.
*   **Reusability:** You can define your templates in one place and import them wherever you need them, making your prompt engineering much more organized.
*   **Advanced Features:** LangChain provides more advanced templates, like `FewShotPromptTemplate`, which makes it easy to include examples in your prompt to guide the model's behavior—a powerful technique for improving accuracy.

## Next Steps

Prompt Templates are the first step in building modular LLM applications. Now that we know how to create dynamic prompts, let's learn how to bundle them together with LLMs into a single, powerful unit called a **Chain**.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Chains](./../05-chains/README.md)
