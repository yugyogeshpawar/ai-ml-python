# 1. LangChain Expression Language (LCEL): Composing Chains Declaratively

Welcome to the first advanced topic: **LangChain Expression Language (LCEL)**. This is a relatively new and incredibly powerful way to compose chains in LangChain. If you've been using `LLMChain` and `SequentialChain`, LCEL offers a more flexible, readable, and performant alternative.

## What is LCEL?

LCEL is a declarative way to build complex chains by "piping" components together. It's inspired by Unix pipes (`|`) and allows you to define a sequence of operations where the output of one step becomes the input of the next.

**Key benefits of LCEL:**
*   **Readability:** Chains built with LCEL are often much easier to read and understand, especially as they grow in complexity.
*   **Flexibility:** It allows for more complex graph-like structures, not just linear sequences. You can easily combine, branch, and merge chains.
*   **Streaming Support:** LCEL is designed from the ground up to support streaming, which is crucial for real-time applications.
*   **Asynchronous Support:** It natively supports `async/await` for improved performance.
*   **Optimized for Production:** LCEL chains are generally more performant and easier to debug in production environments.

## Step-by-Step Code Tutorial

Let's recreate our simple company name generator using LCEL.

### 1. Import Necessary Components

We'll need `ChatOpenAI` (LCEL works best with Chat Models), `PromptTemplate`, and the `StrOutputParser`.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
```

### 2. Create the Script

The `lcel_example.py` script demonstrates how to build a chain with LCEL.

### Key Concepts in the Code

1.  **`prompt = ChatPromptTemplate.from_template(...)`**: We use `ChatPromptTemplate` which is optimized for chat models and LCEL.

2.  **`model = ChatOpenAI(temperature=0.7)`**: We initialize our Chat Model.

3.  **`output_parser = StrOutputParser()`**: This is a simple output parser that converts the `AIMessage` content into a string.

4.  **`chain = prompt | model | output_parser`**: This is the magic of LCEL!
    *   The `|` operator (pipe) connects the components.
    *   `prompt`: Takes the input, formats it into a `ChatPromptValue`.
    *   `model`: Takes the `ChatPromptValue`, sends it to the LLM, and returns an `AIMessage`.
    *   `output_parser`: Takes the `AIMessage` and extracts its string content.

5.  **`response = chain.invoke({"product": "colorful socks"})`**: We use the `.invoke()` method to run the chain. It takes a dictionary of inputs (matching the prompt's input variables) and returns the final output.

## Why LCEL is the Future of LangChain Development

LCEL is the recommended way to build new applications with LangChain. It provides a powerful and intuitive syntax for composing complex LLM applications. It's designed to be highly flexible, allowing you to:
*   **Parallelize steps:** Run multiple components simultaneously.
*   **Add fallbacks:** Define alternative paths if a component fails.
*   **Stream intermediate steps:** Get partial results as the chain executes.

## Next Steps

LCEL is a powerful tool for composing chains. Now that you understand how to build chains declaratively, let's explore how to create your own custom components and integrate them into your LangChain applications.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Custom Chains and Agents](./../02-custom-chains-agents/README.md)
