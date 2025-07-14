# Exercises: Embeddings & Vector Stores

These exercises will help you solidify your understanding of embeddings and vector stores.

### Exercise 1: Experiment with Semantic Similarity

1.  Copy the `embeddings_vector_store_example.py` script.
2.  Add a few more `Document` objects to the `documents` list. Try to include some sentences that are semantically similar to existing ones, and some that are completely unrelated.
    *   Example: `Document(page_content="Cats are independent and love to nap.", metadata={"source": "sentence6"})`
3.  Run `vectorstore.similarity_search()` with different queries.
    *   Try queries that are very similar to one of your documents.
    *   Try queries that are semantically related but don't use the exact same words.
    *   Try queries that are completely unrelated.
4.  Observe the order of the returned documents. Does the most relevant document always appear first? How do the scores (if your vector store provides them) reflect the similarity?

### Exercise 2: Use a Different Embedding Model (Conceptual)

While we're using OpenAI Embeddings, LangChain supports many others (e.g., HuggingFaceEmbeddings, CohereEmbeddings).

1.  Research another embedding model supported by LangChain (e.g., from Hugging Face).
2.  Identify the necessary `pip install` command for that model.
3.  Identify how you would initialize that embedding model in your Python script (e.g., `from langchain.embeddings import HuggingFaceEmbeddings; embeddings = HuggingFaceEmbeddings()`).
4.  Explain, conceptually, how you would swap out `OpenAIEmbeddings` for this new embedding model in your `embeddings_vector_store_example.py` script. You don't need to run the code, just describe the changes.

This exercise highlights LangChain's modularity and the ease of swapping components.

### Exercise 3: Persist a Vector Store

For in-memory vector stores like FAISS, the data is lost when the script finishes. In real applications, you often need to save and load your vector store.

1.  Modify your `embeddings_vector_store_example.py` script.
2.  After creating the `vectorstore` (e.g., `FAISS.from_documents(...)`), use the `vectorstore.save_local("faiss_index")` method to save the index to a local directory named `faiss_index`.
3.  In a *separate* part of your script (or a new script), demonstrate loading the saved index: `loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)`.
    *   **Important:** When loading, you need to pass the `embeddings` model again, as the vector store needs it to convert new queries into embeddings for comparison.
4.  Perform a `similarity_search` on the `loaded_vectorstore` to confirm it works.

This exercise is crucial for understanding how to manage vector stores in persistent applications.
