# 5. Chains in LangChain: Combining Components into Workflows

We've learned about LLMs and Prompt Templates. Now, we'll learn how to combine them using **Chains**. This is the "aha!" moment where the name "LangChain" really starts to make sense. Chains are the fundamental way to build structured, multi-step workflows.

## What Exactly is a Chain?

A Chain, in its simplest form, is a pipeline that processes an input and produces an output. It "chains" together different components, where the output of one component becomes the input for the next.

The most basic and common chain is the `LLMChain`. It represents a single, fundamental operation in an LLM application:
1.  Take user input.
2.  Use a `PromptTemplate` to format the input into a prompt.
3.  Pass the formatted prompt to an `LLM`.
4.  Return the LLM's output.

**Analogy: A Chain is like an assembly line.**
-   **Input (e.g., a product name):** The raw materials at the start of the line.
-   **Prompt Template:** The first station, which puts the raw materials into a specific mold.
-   **LLM:** The second station, which performs an action (like painting or shaping) on the molded material.
-   **Output (e.g., a company name):** The finished product at the end of the line.

By creating a chain, you encapsulate this entire assembly line into a single, reusable object.

## Step-by-Step Code Tutorial

The `llm_chain_example.py` script shows how to build and use an `LLMChain`. Let's analyze it.

### 1. Import `LLMChain`

```python
from langchain.chains import LLMChain
```
Alongside our `OpenAI` and `PromptTemplate` imports, we now import the `LLMChain` class.

### 2. Set up the Components

First, we define the individual components that will make up our chain: the LLM and the Prompt Template. This is exactly the same as in the previous lessons.

```python
llm = OpenAI(temperature=0.6)
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

### 3. Create the `LLMChain` Instance

This is where we assemble our pipeline. We create an instance of `LLMChain` and pass our components to its constructor.

```python
name_chain = LLMChain(llm=llm, prompt=prompt_template)
```
We've now created a `name_chain` object that knows how to take a `product`, use our `prompt_template` to create a prompt, and then call our `llm` with that prompt.

### 4. Run the Chain

This is the beautiful part. Instead of manually formatting the prompt and calling the LLM, we can now just use the chain's `.run()` method.

```python
response = name_chain.run("artisanal coffee beans")
```
The `.run()` method is a convenient shortcut that handles the entire process for us. It takes the input variable (`product="artisanal coffee beans"`), executes the chain's logic, and returns the final string output from the LLM.

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
