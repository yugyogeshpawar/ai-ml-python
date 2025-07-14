# 4. Retrievers: The Engine of Information Fetching

In the last lesson, we created a vector store and performed a simple `similarity_search`. While this is useful, LangChain provides a more powerful and standardized abstraction for fetching documents: the **Retriever**.

## What is a Retriever?

A Retriever is an interface that is responsible for fetching relevant `Document` objects based on a query. It's a more general concept than a vector store. While the most common type of retriever is one that is backed by a vector store, other types of retrievers exist (e.g., retrievers that query a traditional database or a search engine).

The key method of a retriever is `get_relevant_documents()`, which takes a query string and returns a list of relevant `Document` objects.

## Why Use a Retriever Instead of Directly Using the Vector Store?

Using the retriever interface provides several advantages:

1.  **Standardization:** It provides a consistent, unified interface for fetching documents, regardless of the underlying method. This means you can easily swap out a vector store retriever for another type of retriever without changing the rest of your chain's code.
2.  **Advanced Retrieval Strategies:** The retriever interface allows for more sophisticated retrieval methods beyond simple similarity search. For example:
    *   **Maximum Marginal Relevance (MMR):** This method fetches documents that are not only relevant to the query but also diverse, avoiding a set of results that are all very similar to each other.
    *   **Self-Querying:** A retriever that can use an LLM to write its own metadata filters based on the user's query.
    *   **Contextual Compression:** A retriever that can compress the retrieved documents to only include the most relevant parts, saving tokens and reducing noise.
3.  **Chain Integration:** Retrievers are designed to be seamlessly integrated into LangChain's `RetrievalQA` chain (and others), which is the standard way to build RAG applications.

## Step-by-Step Code Tutorial

Let's see how to convert our vector store from the previous lesson into a retriever and use it.

### 1. Build the Vector Store

First, we need to create our vector store, just as we did before. This involves loading documents, splitting them (if necessary), and embedding them into a vector store like FAISS.

### 2. Create the Script

The `retriever_example.py` script demonstrates how to create and use a retriever.

### Key Concepts in the Code

1.  **`vectorstore = FAISS.from_documents(documents, embeddings)`**: We create our vector store as usual.

2.  **`retriever = vectorstore.as_retriever()`**: This is the key step. The `.as_retriever()` method converts the vector store into a retriever object that follows the standard retriever interface.

3.  **`retrieved_docs = retriever.get_relevant_documents(query)`**: We use the retriever's standard method, `get_relevant_documents()`, to fetch the documents. Under the hood, this is still performing a similarity search on the vector store, but we are now using the standardized retriever interface.

### Configuring the Retriever

You can configure the retriever to use different search strategies. For example, to specify that you only want the top 2 most relevant documents (`k=2`):

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

To use Maximum Marginal Relevance (MMR) for more diverse results:

```python
retriever = vectorstore.as_retriever(search_type="mmr")
```

## Next Steps

We have now completed the entire "retrieval" part of RAG:
1.  **Load:** Using Document Loaders.
2.  **Split:** Using Text Splitters.
3.  **Embed & Store:** Using Embedding Models and Vector Stores.
4.  **Retrieve:** Using Retrievers.

The final step is to "augment" our generation. We need to take the documents retrieved by our retriever and "stuff" them into a prompt, along with the user's question, and send it to an LLM to generate the final answer. This is typically done with a `RetrievalQA` chain, which we will explore in the project for this part.

But before we build the final RAG chain, let's explore another crucial component for building interactive applications: **Memory**.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Memory](./../05-memory/README.md)
