# Exercises: Prompt Templates

These exercises will help you master the art of creating dynamic and effective prompts.

### Exercise 1: Create a Multi-Variable Prompt Template

1.  Create a new Python script.
2.  Define a `PromptTemplate` that takes two input variables: `style` and `topic`.
3.  The template should ask the LLM to generate a short poem in a specific `style` about a given `topic`. For example: "Write a haiku about the topic: {topic}."
4.  Format the prompt with the `style` "limerick" and the `topic` "a robot that loves to cook."
5.  Pass the formatted prompt to an LLM and print the result.

### Exercise 2: Design a Few-Shot Prompt

You want to build a system that can extract the main programming language from a job description. To ensure the LLM provides a consistent, one-word answer, you decide to use a few-shot prompt.

1.  Create a list of `examples`. Each example should be a dictionary with a "description" and the corresponding "language".
    *   Example 1: Description about a data science role -> "Python"
    *   Example 2: Description about a front-end role -> "JavaScript"
    *   Example 3: Description about a mobile dev role -> "Swift"
2.  Create an `example_template` to format these examples.
3.  Use `FewShotPromptTemplate` to combine the examples with a prefix, a suffix, and the example template. The prefix should explain the task (e.g., "Extract the primary programming language from the job description."). The suffix should contain the user's input.
4.  Format the `FewShotPromptTemplate` with a new job description and print the final prompt. You don't need to run this with an LLM, just observe the final prompt structure.

### Exercise 3: Refactor a Prompt

Imagine you have the following hard-coded prompt in your application:

```python
user_name = "Alex"
question = "the capital of France"

prompt = f"""
Hello {user_name}, you asked me to find out about {question}.
Here is the information I found:
The capital of France is Paris.
Could you please ask another question?
"""
```

This prompt is not reusable. Your task is to refactor this into a `PromptTemplate`.

1.  Identify the variables in the prompt (`user_name`, `question`).
2.  Create a `PromptTemplate` that uses these variables.
3.  Format the template with the example `user_name` and `question` to verify that it produces the same output.
