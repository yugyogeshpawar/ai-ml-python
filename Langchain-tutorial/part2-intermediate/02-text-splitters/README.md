# 2. Text Splitters: Managing Large Documents for LLMs

In the previous lesson, we learned how to load documents using Document Loaders. However, a common challenge with LLMs is their **context window limit**. This means that models can only process a certain amount of text at a time. If your document is too large, you can't simply pass the entire `page_content` to the LLM.

This is where **Text Splitters** come into play.

## What are Text Splitters?

A Text Splitter is a utility that takes a long string of text (or a list of `Document` objects) and breaks it down into smaller, manageable chunks. These chunks are designed to fit within an LLM's context window while retaining as much semantic meaning as possible.

The goal of a good text splitter is to:
1.  **Break text into small enough chunks:** So they fit within the LLM's token limit.
2.  **Maintain semantic coherence:** Avoid splitting in the middle of a sentence, paragraph, or code block if possible, to keep related information together.
3.  **Add overlap (optional but recommended):** Include a small overlap between chunks to ensure that context isn't lost at the boundaries.

LangChain offers various types of text splitters, each with different strategies for splitting text. The most commonly used is `RecursiveCharacterTextSplitter`.

## Step-by-Step Code Tutorial

Let's demonstrate how to use `RecursiveCharacterTextSplitter` to break down a long piece of text.

### 1. Create a Sample Document

First, let's create a long string of text that we want to split. For a real application, this would come from a Document Loader.

```python
long_text = """
LangChain is a framework for developing applications powered by language models. It enables applications that:
1. Are data-aware: connect a language model to other sources of data.
2. Are agentic: allow a language model to interact with its environment.

The main components of LangChain are:
- Models: The various model types LangChain supports (LLMs, ChatModels, Text Embedding Models).
- Prompts: Prompt management, optimization, and serialization.
- Chains: Composable sequences of calls (to LLMs or other utilities).
- Indexes: Ways to structure documents and interact with them.
- Memory: Persist application state between runs of a chain/agent.
- Agents: LLMs that make decisions about which Actions to take, take Action, observe results, and repeat.

LangChain is designed to be modular and extensible, allowing developers to easily swap out components and build custom solutions. It supports various integrations with different LLM providers, vector stores, and tools.
"""
```

### 2. Create the Script

The `text_splitter_example.py` script shows how to use the splitter.

### Key Concepts in the Code

1.  **`from langchain.text_splitter import RecursiveCharacterTextSplitter`**: We import the splitter class.

2.  **`text_splitter = RecursiveCharacterTextSplitter(...)`**: We initialize the splitter with key parameters:
    *   **`chunk_size`**: The maximum size of each chunk (in characters or tokens, depending on the splitter).
    *   **`chunk_overlap`**: The number of characters (or tokens) that overlap between consecutive chunks. This helps maintain context across splits. A common practice is to set `chunk_overlap` to about 10-20% of `chunk_size`.
    *   **`separators`**: (For `RecursiveCharacterTextSplitter`) A list of characters to try splitting on, in order of preference. It tries to split on larger units first (e.g., `\n\n` for paragraphs), then smaller units (e.g., `\n` for lines), and finally individual characters. This helps maintain semantic boundaries.

3.  **`chunks = text_splitter.create_documents([long_text])`**: This method takes a list of strings (or `Document` objects) and returns a list of new `Document` objects, where each `Document` represents a chunk. The original metadata is preserved and copied to the new chunks.

## Why Use Text Splitters?

*   **Overcome Context Window Limits:** Essential for processing documents larger than what an LLM can handle in a single prompt.
*   **Improved Relevance:** By breaking text into semantically meaningful chunks, you ensure that when you retrieve information later, you get relevant snippets rather than incomplete sentences.
*   **Cost Efficiency:** Sending smaller chunks to the LLM can be more cost-effective, as you only pay for the tokens you use.

## Next Steps

Once we have our documents split into manageable chunks, the next challenge is to efficiently find the most relevant chunks when a user asks a question. This involves converting our text into numerical representations (embeddings) and storing them in a specialized database (a vector store).

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Embeddings & Vector Stores](./../03-embeddings-vector-stores/README.md)
