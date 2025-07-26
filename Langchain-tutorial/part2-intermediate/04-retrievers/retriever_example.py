from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Ensure your OPENAI_API_KEY is set as an environment variable.

# 1. Create Sample Documents and Vector Store (same as previous lesson)
documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "sentence1"}),
    Document(page_content="A dog is a man's best friend.", metadata={"source": "sentence2"}),
    Document(page_content="The cat sat on the mat.", metadata={"source": "sentence3"}),
    Document(page_content="Foxes are known for their cunning.", metadata={"source": "sentence4"}),
    Document(page_content="The dog chased the cat.", metadata={"source": "sentence5"}),
]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 2. Convert the Vector Store into a Retriever
# The as_retriever() method provides a standardized interface for fetching documents.
# We can configure it with search parameters, like `k` for the number of documents to return.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Use the Retriever to get relevant documents
# The standard method for all retrievers is get_relevant_documents().
query = "What are canines known for?"
retrieved_docs = retriever.get_relevant_documents(query)

# Print the results
print(f"--- Retrieved documents for query: '{query}' ---")
for i, doc in enumerate(retrieved_docs):
    print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
    print(doc.page_content)

# --- Example of using a different search type ---

# Create a retriever that uses Maximum Marginal Relevance (MMR)
# MMR tries to find a set of documents that are both relevant to the query and diverse.
mmr_retriever = vectorstore.as_retriever(search_type="mmr")

query_mmr = "Tell me about a pet."
retrieved_docs_mmr = mmr_retriever.get_relevant_documents(query_mmr)

print(f"\n--- MMR Retrieved documents for query: '{query_mmr}' ---")
for i, doc in enumerate(retrieved_docs_mmr):
    print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
    print(doc.page_content)
