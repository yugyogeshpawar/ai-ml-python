# Exercises: Memory

These exercises will help you explore different types of memory and their impact on conversational chains.

### Exercise 1: Experiment with Different Memory Types

LangChain offers various memory implementations. Let's compare two: `ConversationBufferMemory` and `ConversationSummaryMemory`.

1.  Copy the `memory_example.py` script.
2.  Modify the script to use `ConversationSummaryMemory` instead of `ConversationBufferMemory`. You'll need to install `pip install langchain-openai`.
3.  Run the script and have a conversation with the bot. Ask it several questions, building up a longer history.
4.  Compare the behavior of the two memory types.
    *   Does `ConversationBufferMemory` simply store the entire conversation verbatim?
    *   Does `ConversationSummaryMemory` try to condense the conversation into a summary? How does this affect the bot's ability to recall specific details vs. the overall context?
    *   What happens when the conversation gets very long? Does `ConversationBufferMemory` start to truncate the history?

### Exercise 2: Create a Memory-Aware Chain with Metadata

1.  Combine concepts from this lesson and the previous one on Retrievers.
2.  Create a chain that:
    *   Loads documents from a source (e.g., a website using `WebBaseLoader`).
    *   Splits the documents into chunks.
    *   Embeds the chunks and stores them in a vector store.
    *   Creates a retriever from the vector store.
    *   **Adds `ConversationBufferMemory` to the chain.**
3.  Modify the prompt template to include both the `chat_history` and the retrieved documents as context.
4.  Now, ask the chain questions that require it to use both the retrieved information and the previous conversation history.

This exercise demonstrates how to build a truly powerful conversational agent that can access external knowledge and remember past interactions.

### Exercise 3: Explore Contextual Compression

1.  Research `ContextualCompressionRetriever` in LangChain. This retriever attempts to compress the retrieved documents to only include the most relevant parts, saving tokens and reducing noise.
2.  Identify the necessary steps to implement this retriever in your code. This might involve installing additional packages and modifying your retriever setup.
3.  Explain, conceptually, how this retriever could improve the performance of a RAG system compared to a standard retriever that simply returns the top-k documents.
