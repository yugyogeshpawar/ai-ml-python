from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Create a long string of text to demonstrate splitting
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

# Initialize the RecursiveCharacterTextSplitter
# chunk_size: The maximum number of characters in each chunk.
# chunk_overlap: The number of characters that overlap between consecutive chunks.
# separators: A list of characters to try splitting on, in order of preference.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

# Create Document objects from the long text
# The splitter can take a list of strings or a list of Document objects.
# It returns a list of new Document objects, each representing a chunk.
chunks = text_splitter.create_documents([long_text])

# Print the chunks to see the result
print(f"Original text length: {len(long_text)} characters")
print(f"Number of chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} (Length: {len(chunk.page_content)}) ---")
    print(chunk.page_content)
    print(f"Metadata: {chunk.metadata}") # Metadata from original document is preserved
