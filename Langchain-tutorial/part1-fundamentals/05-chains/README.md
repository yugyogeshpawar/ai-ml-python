# 5. Chains in LangChain: Combining Components into Workflows

We've learned about LLMs and Prompt Templates. Now, we'll learn how to combine them using **Chains**. This is the "aha!" moment where the name "LangChain" really starts to make sense. Chains are the fundamental way to build structured, multi-step workflows.

## What Exactly is a Chain?

A Chain is a pipeline that processes an input and produces an output. It "chains" together different components, where the output of one component becomes the input for the next. With the introduction of the **LangChain Expression Language (LCEL)**, creating chains has become more intuitive and powerful.

The most basic chain combines a `PromptTemplate` and an `LLM`. It represents a single, fundamental operation:
1.  Take user input.
2.  Use a `PromptTemplate` to format the input into a prompt.
3.  Pass the formatted prompt to an `LLM`.
4.  Return the LLM's output.

**Analogy: A Chain is like an assembly line.**
-   **Input (e.g., a product name):** The raw materials at the start of the line.
-   **Prompt Template:** The first station, which puts the raw materials into a specific mold.
-   **LLM:** The second station, which performs an action (like painting or shaping) on the molded material.
-   **Output (e.g., a company name):** The finished product at the end of the line.

LCEL allows us to define this assembly line using the pipe (`|`) operator, making the flow of data explicit and readable.

## Step-by-Step Code Tutorial

The `llm_chain_example.py` script shows how to build a chain using LCEL. Let's analyze it.

### 1. Set up the Components

First, we define the individual components that will make up our chain: the LLM and the Prompt Template. This is exactly the same as in the previous lessons.

```python
llm = OpenAI(temperature=0.6)
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

### 2. Create the Chain with LCEL

This is where the magic happens. We use the pipe (`|`) operator to connect our components.

```python
name_chain = prompt_template | llm
```
This single line of code creates a chain that:
1.  First, takes an input dictionary (e.g., `{"product": "artisanal coffee beans"}`).
2.  Pipes it to the `prompt_template`, which formats it into a full prompt string.
3.  Pipes the resulting prompt string to the `llm`, which generates the final output.

### 3. Run the Chain

Instead of a `.run()` method, LCEL chains use `.invoke()`.

```python
response = name_chain.invoke({"product": "artisanal coffee beans"})
```
The `.invoke()` method takes a dictionary where the keys match the input variables of the first component in the chain (in this case, the `prompt_template`). It executes the entire pipeline and returns the final output.

## The Power of Chains

Why go to the trouble of creating a chain object?

*   **Simplicity and Readability:** Your main application logic becomes much cleaner. Instead of multiple lines for formatting and predicting, you have a single, descriptive line: `name_chain.run(product)`.
*   **Encapsulation:** The chain bundles a specific capability (like "generating a company name") into a single object. This makes your code more organized and easier to reason about.
*   **Composability (The Real Magic):** `LLMChain` is just the beginning. LangChain's true power is revealed when you start chaining chains together. You can create a `SequentialChain` where the output of one `LLMChain` becomes the input for another. For example:
    1.  **Chain 1:** Takes a product, outputs a company name.
    2.  **Chain 2:** Takes the company name, outputs a slogan for that company.
    A `SequentialChain` would manage this entire two-step process.

## Next Steps

Our chain now produces a company name, but it's just a raw string. What if we wanted to guarantee the output was in a specific format, like a JSON object or a Python list? To do that, we need to add one more component to our pipeline: an **Output Parser**.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Output Parsers](./../06-output-parsers/README.md)
