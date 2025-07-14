# 3. Embeddings & Vector Stores: The Foundation of Semantic Search

In the previous lessons, we learned how to load documents and split them into manageable chunks. Now, we face a critical question: how do we efficiently find the *most relevant* chunks when a user asks a question? Simply searching for keywords isn't enough; we need to understand the *meaning* of the text.

This is where **Embeddings** and **Vector Stores** come into play. They are the core components that enable **semantic search** and **Retrieval-Augmented Generation (RAG)**.

## What are Embeddings? A Deeper, Simpler Look

Imagine you have a library with thousands of books. If you wanted to find a book about "brave knights," you could search for the keyword "knight." But what if the perfect book doesn't use the word "knight" and instead talks about "chivalrous warriors" or "gallant heroes"? A keyword search would miss it entirely.

This is the problem embeddings solve for language. An **Embedding** is a way to translate the meaning of a word, sentence, or document into a list of numbers, called a **vector**.

**Analogy: The Universal Language of Meaning**
Think of embeddings as a universal translator that converts every piece of text into a specific set of coordinates on a giant map of meaning.
-   The sentence "The king ruled the land" would get a set of coordinates.
-   The sentence "The monarch governed the country" would get a *very similar* set of coordinates, because it means almost the same thing.
-   The sentence "The chef cooked a meal" would get a set of coordinates that are very far away from the first two.

**Key characteristics of embeddings:**
*   **Semantic Meaning:** The magic of embeddings is that text with similar meanings will have similar numerical representations (vectors) and thus will be "close" to each other in the vector space. This allows us to find documents based on their meaning, not just their keywords.
*   **High-Dimensional:** This "map of meaning" isn't 2D or 3D. It has hundreds or even thousands of dimensions, allowing for very nuanced representations of meaning.
*   **Created by Embedding Models:** Specialized neural networks (embedding models) are trained on vast amounts of text to learn these relationships between words and concepts. LangChain provides interfaces to various embedding models (e.g., OpenAI Embeddings, Hugging Face Embeddings).

## What are Vector Stores?

A **Vector Store** (also known as a vector database) is a specialized database designed to store and search these high-dimensional embedding vectors. Its primary function is to perform **similarity search**: given the coordinates (vector) of a query, it can instantly find the document vectors with the closest coordinates on the map.

**How it works:**
1.  You take your document chunks (from the text splitter).
2.  You use an embedding model to convert each chunk into an embedding vector (its coordinates on the map of meaning).
3.  You store these embedding vectors (along with their original text and metadata) in a vector store.
4.  When a user asks a question, you convert their question into an embedding vector.
5.  You then query the vector store with this question embedding.
6.  The vector store returns the document chunks whose embeddings are most "similar" (closest in the vector space) to the query embedding.

LangChain integrates with many popular vector stores like FAISS, Chroma, Pinecone, Weaviate, and more. For local development and simple use cases, FAISS and Chroma are excellent choices.

## Step-by-Step Code Tutorial

Let's demonstrate how to create embeddings and store them in a simple in-memory vector store (FAISS).

### 1. Install Necessary Libraries

We'll need `tiktoken` for token counting (used by OpenAI embeddings) and `faiss-cpu` for the FAISS vector store.

```bash
pip install tiktoken faiss-cpu
```

### 2. Create Sample Documents

For this example, we'll use a few simple documents. In a real application, these would come from your Document Loader and Text Splitter.

```python
from langchain.schema import Document

documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "sentence1"}),
    Document(page_content="A dog is a man's best friend.", metadata={"source": "sentence2"}),
    Document(page_content="The cat sat on the mat.", metadata={"source": "sentence3"}),
    Document(page_content="Foxes are known for their cunning.", metadata={"source": "sentence4"}),
]
```

### 3. Create the Script

The `embeddings_vector_store_example.py` script shows how to do this.

### Key Concepts in the Code

1.  **`from langchain_openai import OpenAIEmbeddings`**: We import the embedding model. `OpenAIEmbeddings` uses OpenAI's text embedding API.

2.  **`from langchain.vectorstores import FAISS`**: We import the FAISS vector store.

3.  **`embeddings = OpenAIEmbeddings()`**: We initialize our embedding model. This object will convert text into vectors.

4.  **`vectorstore = FAISS.from_documents(documents, embeddings)`**: This is the core step.
    *   `FAISS.from_documents()`: This class method takes a list of `Document` objects and an `embeddings` model.
    *   It iterates through each document, uses the `embeddings` model to convert its `page_content` into a vector, and then stores that vector (along with the original `Document` object) in the FAISS index.

5.  **`query = "What animal is known for being lazy?"`**: We define a query.

6.  **`docs = vectorstore.similarity_search(query)`**: This is how we perform a semantic search.
    *   The `similarity_search()` method takes our query string.
    *   It first uses the same `embeddings` model to convert the query string into a query vector.
    *   Then, it searches its internal index for the document embeddings that are most similar (closest) to the query vector.
    *   It returns a list of the most relevant `Document` objects.

## Why are Embeddings and Vector Stores Crucial for LLM Applications?

*   **Semantic Understanding:** They allow your LLM application to understand the *meaning* of text, not just keywords. This is fundamental for building intelligent Q&A systems.
*   **Retrieval-Augmented Generation (RAG):** This is the backbone of RAG. Instead of relying solely on the LLM's pre-trained knowledge (which can be outdated or hallucinate), you can retrieve relevant, up-to-date information from your own data and provide it to the LLM as context. This significantly improves accuracy and reduces hallucinations.
*   **Scalability:** Vector stores are designed for efficient similarity search over millions or billions of vectors, making them scalable for large datasets.

## Next Steps

Now that we can load, split, embed, and store our documents, the next logical step is to learn how to *retrieve* the most relevant information from our vector store when a user asks a question. This is the role of **Retrievers**.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Retrievers](./../04-retrievers/README.md)
