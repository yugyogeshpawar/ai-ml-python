# Exercises: Retrievers

These exercises will help you understand how to configure and use retrievers effectively.

### Exercise 1: Configure Search `k`

1.  Copy the `retriever_example.py` script.
2.  Modify the retriever so that it only returns the single most relevant document (i.e., the top 1).
3.  Run a query and verify that you only get one document back.
4.  Now, modify the retriever to return the top 4 documents. Run the same query and observe the results.

This exercise demonstrates how to control the number of documents retrieved, which is a key parameter for balancing context length and cost.

### Exercise 2: Use Metadata Filtering

Many vector stores support filtering based on the metadata of your documents. This is a powerful way to narrow down the search space.

1.  In your `retriever_example.py` script, add more `Document` objects with different metadata. For example, add a `category` field:
    ```python
    Document(page_content="Lions are the kings of the jungle.", metadata={"source": "sentence6", "category": "feline"})
    Document(page_content="Golden retrievers are friendly dogs.", metadata={"source": "sentence7", "category": "canine"})
    ```
2.  Recreate your vector store with this new list of documents.
3.  Now, create a retriever that is configured to *only* search for documents where the `category` is `"canine"`. The syntax for this can vary between vector stores, but for FAISS and many others, you can use the `filter` argument in the `similarity_search` method (or configure it in `as_retriever`).
    *   **Hint:** For this exercise, it's easier to apply the filter during the search: `vectorstore.similarity_search(query, filter={"category": "canine"})`.
4.  Run a general query like "Tell me about an animal." and verify that you only get back documents from the "canine" category, even if a "feline" document is semantically similar.

This exercise shows how to combine semantic search with traditional metadata filtering for more precise results.

### Exercise 3: Compare Search Types

1.  Use the `retriever_example.py` script.
2.  Define a query that is likely to have several similar results, like "Tell me about a pet."
3.  Run the query using the default retriever (`search_type="similarity"`) and print the results.
4.  Now, run the same query using the MMR retriever (`search_type="mmr"`) and print the results.
5.  Compare the two sets of results. Is the MMR result set more diverse? Does it include documents that the default search might have overlooked in favor of multiple, very similar results?

This exercise helps you understand the trade-offs between pure similarity and result diversity.
