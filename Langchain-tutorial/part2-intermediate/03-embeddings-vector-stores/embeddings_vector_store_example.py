from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 1. Create Sample Documents
documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "sentence1"}),
    Document(page_content="A dog is a man's best friend.", metadata={"source": "sentence2"}),
    Document(page_content="The cat sat on the mat.", metadata={"source": "sentence3"}),
    Document(page_content="Foxes are known for their cunning.", metadata={"source": "sentence4"}),
    Document(page_content="The dog chased the cat.", metadata={"source": "sentence5"}),
]

# 2. Initialize the Local Embedding Model
# This uses a model from Hugging Face that runs on your machine.
# The first time you run this, it will download the model (approx. 90MB).
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create a Vector Store from the documents and embeddings
vectorstore = FAISS.from_documents(documents, embeddings)

print("Vector store created successfully with documents using a local model.")

# 4. Perform a Similarity Search
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
