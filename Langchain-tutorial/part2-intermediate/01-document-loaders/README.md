# 1. Document Loaders: Ingesting Your Data

The first step in building a data-aware application is to get your data into a format that LangChain can work with. This is the job of **Document Loaders**.

## What are Document Loaders?

A Document Loader is a specialized tool designed to load data from a source and convert it into a list of `Document` objects. A `Document` is a simple LangChain object that contains two things:
-   `page_content`: A string containing the actual text of the document.
-   `metadata`: A dictionary containing information about the document, such as its source, page number, or other attributes.

LangChain has a huge ecosystem of document loaders for a wide variety of sources, including:
-   Text files (`.txt`)
-   PDFs (`.pdf`)
-   Web pages (HTML)
-   CSV files
-   JSON files
-   Databases (SQL)
-   And many more!

## Step-by-Step Code Tutorial

Let's see how to use a `WebBaseLoader` to load the content from a live web page.

### 1. Install the Necessary Library

To load web pages, we need to install the `beautifulsoup4` library, which is excellent for parsing HTML.

```bash
pip install beautifulsoup4
```

### 2. Create the Script

The `document_loader_example.py` script shows how to do this.

### Key Concepts in the Code

1.  **`from langchain_community.document_loaders import WebBaseLoader`**: We import the specific loader we want to use from the `langchain_community` package. If we were loading a PDF, we might import `PyPDFLoader` from the same package.

2.  **`loader = WebBaseLoader("https://example.com")`**: We create an instance of the loader, passing in the source of our data. In this case, it's the URL of the website we want to load.

3.  **`documents = loader.load()`**: This is the core method. The `.load()` method performs the action of fetching the data (in this case, making an HTTP request to the URL), processing it, and returning a list of `Document` objects.

4.  **Inspecting the Output**: The script then iterates through the loaded documents and prints their `page_content` and `metadata`. For the `WebBaseLoader`, the metadata typically includes the `source` (the URL) and sometimes the `title` of the web page.

## Why Use Document Loaders?

Using a dedicated loader abstracts away the complexity of data ingestion. You don't have to worry about opening files, making HTTP requests, or parsing complex formats like PDF. You simply choose the right loader for your source, call `.load()`, and you get a clean, standardized list of `Document` objects, ready for the next step in your pipeline.

## Next Steps

Now that we have our documents loaded, we face a new problem. LLMs have a limited "context window," meaning they can only process a certain amount of text at a time. If our loaded document is very long (like a book or a long research paper), we can't just stuff it all into a single prompt.

In the next lesson, we'll learn how to solve this problem using **Text Splitters**.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Text Splitters](./../02-text-splitters/README.md)
