from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Ensure your OPENAI_API_KEY is set as an environment variable.

# 1. Create Sample Documents
# In a real application, these would come from Document Loaders and Text Splitters.
documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "sentence1"}),
    Document(page_content="A dog is a man's best friend.", metadata={"source": "sentence2"}),
    Document(page_content="The cat sat on the mat.", metadata={"source": "sentence3"}),
    Document(page_content="Foxes are known for their cunning.", metadata={"source": "sentence4"}),
    Document(page_content="The dog chased the cat.", metadata={"source": "sentence5"}),
]

# 2. Initialize the Embedding Model
# This model converts text into numerical vectors (embeddings).
embeddings = OpenAIEmbeddings()

# 3. Create a Vector Store (FAISS in this case) from the documents and embeddings
# FAISS.from_documents takes your documents and the embedding model,
# converts each document's content into an embedding, and stores it.
vectorstore = FAISS.from_documents(documents, embeddings)

print("Vector store created successfully with documents.")

# 4. Perform a Similarity Search
# When you query, your query string is also converted into an embedding,
# and the vector store finds the most similar document embeddings.
query = "What animal is known for being lazy?"
docs = vectorstore.similarity_search(query)

print(f"\n--- Similarity Search Results for: '{query}' ---")
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
    print(doc.page_content)

# Another query
query_2 = "Tell me about a feline."
docs_2 = vectorstore.similarity_search(query_2)

print(f"\n--- Similarity Search Results for: '{query_2}' ---")
for i, doc in enumerate(docs_2):
    print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
    print(doc.page_content)
