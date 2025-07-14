# Exercises: Text Splitters

These exercises will help you understand how to effectively split documents for LLMs.

### Exercise 1: Experiment with `chunk_size` and `chunk_overlap`

1.  Copy the `text_splitter_example.py` script.
2.  Modify the `long_text` variable to be a paragraph from a book or a long article (at least 500 characters).
3.  Run the script with `chunk_size=100` and `chunk_overlap=0`. Observe the number of chunks and their content. Do any sentences get cut off mid-way?
4.  Now, change `chunk_overlap` to `20` (keeping `chunk_size=100`). Rerun and observe the chunks. How does the overlap affect the content of consecutive chunks?
5.  Finally, increase `chunk_size` to `300` and `chunk_overlap` to `50`. How does this change the number and size of chunks?

This exercise demonstrates the practical impact of these parameters on your document chunks.

### Exercise 2: Use a Different Separator Strategy

The `RecursiveCharacterTextSplitter` uses a list of separators.

1.  Consider a document that is primarily a list of items, where each item is on a new line, but there are no double newlines.
    ```
    Item 1: This is the first item.
    Item 2: This is the second item.
    Item 3: This is the third item.
    ```
2.  If you use the default `separators=["\n\n", "\n", " ", ""]`, how would the splitter behave?
3.  Modify the `text_splitter_example.py` script to use a `long_text` like the one above.
4.  Experiment with changing the `separators` list to prioritize splitting on `"\n"` first, then `""` (empty string for character-level split). Observe the difference in the chunks.

This exercise helps you understand how to tailor the splitter to the structure of your specific documents.

### Exercise 3: Split a List of Documents

The `create_documents` method can take a list of `Document` objects.

1.  Create two simple `Document` objects manually:
    *   `doc1 = Document(page_content="This is the first part of a story. It talks about a hero's journey.", metadata={"source": "story_part1.txt"})`
    *   `doc2 = Document(page_content="The hero faced many challenges. Eventually, they found their way home.", metadata={"source": "story_part2.txt"})`
2.  Create a `RecursiveCharacterTextSplitter` with `chunk_size=50` and `chunk_overlap=10`.
3.  Use `text_splitter.split_documents([doc1, doc2])` to split these documents.
4.  Print the `page_content` and `metadata` of each resulting chunk. Notice how the original metadata is preserved for each chunk.

This exercise demonstrates how text splitters work seamlessly with the `Document` objects produced by Document Loaders.
