# Exercises: Document Loaders

These exercises will help you get hands-on experience with different types of document loaders.

### Exercise 1: Load a Local Text File

1.  Create a new text file in this directory named `my_story.txt`.
2.  Write a short story or a few paragraphs of text inside it.
3.  Create a Python script that uses the `TextLoader` from `langchain.document_loaders` to load `my_story.txt`.
4.  Print the `page_content` and `metadata` of the loaded document. What information is in the metadata?

### Exercise 2: Load a PDF

1.  Find a PDF file on your computer or download one from the internet (e.g., a research paper from arXiv).
2.  Install the necessary library: `pip install pypdf`.
3.  Create a Python script that uses the `PyPDFLoader` to load the PDF.
4.  The `PyPDFLoader` loads each page as a separate `Document`. Iterate through the list of loaded documents and print the page number (from the metadata) and the first 100 characters of each page's content.

### Exercise 3: Explore a Different Loader

1.  Browse the official LangChain documentation for [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/).
2.  Choose a loader that interests you and that works with a data source you have access to (e.g., `CSVLoader`, `JSONLoader`, or even a loader for a specific service like `NotionDirectoryLoader` if you use Notion).
3.  Follow the documentation to install any required packages and write a script to load data from that source.
4.  Inspect the resulting `Document` objects. Pay close attention to the `metadata`. How does the metadata from this loader differ from the others you've used? This exercise highlights the diversity of the LangChain ecosystem.
