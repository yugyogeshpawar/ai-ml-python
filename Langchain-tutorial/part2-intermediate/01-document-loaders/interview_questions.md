# Interview Questions: Document Loaders

### Q1: What is the primary role of a Document Loader in LangChain?

**Answer:**

The primary role of a Document Loader is to **ingest data from a source and convert it into a standardized format** that the rest of the LangChain framework can use. It acts as the entry point for your data, abstracting away the complexities of reading from different file formats or data sources. The output of a loader is always a list of `Document` objects, each containing `page_content` (the text) and `metadata`.

### Q2: What is a `Document` object in LangChain, and what are its key attributes?

**Answer:**

A `Document` object is the standard data structure that LangChain uses to represent a piece of text. It's a simple container with two key attributes:

1.  **`page_content`**: A string that holds the actual text content of the document.
2.  **`metadata`**: A Python dictionary that contains information *about* the document. This is crucial for filtering, tracking, and providing context. The metadata can include things like the source file path, the page number of a PDF, a web URL, or any other custom information you want to associate with the text.

### Q3: You need to load data from a source that LangChain doesn't have a pre-built loader for (e.g., a proprietary API). What would be your approach?

**Answer:**

This is a common scenario, and LangChain is designed to be extensible. The best approach would be to **create a custom Document Loader**.

The process involves:
1.  Creating a new Python class that inherits from LangChain's `BaseLoader`.
2.  Implementing a `load()` method within this class.
3.  Inside the `load()` method, you would write the specific Python code to:
    *   Connect to the proprietary API.
    *   Fetch the data.
    *   Process the raw data and extract the text content and any relevant metadata.
    *   Instantiate one or more `Document` objects with the extracted `page_content` and `metadata`.
    *   Return the list of `Document` objects.

By following this pattern, your custom loader will integrate seamlessly with the rest of the LangChain ecosystem, allowing you to use it with text splitters, vector stores, and retrievers just like any of the built-in loaders.
