import os
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# Ensure your OPENAI_API_KEY is set as an environment variable

# 1. Load the PDF
loader = PyPDFLoader("your_document.pdf")  # Replace with your PDF file
documents = loader.load()

# 2. Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and store them in a VectorStore
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# 4. Create a Retriever
retriever = db.as_retriever()

# 5. Create a Chain to answer questions
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",  # "stuff" method simply concatenates all retrieved docs
    retriever=retriever,
    return_source_documents=True # Returns the docs used to answer
)

# 6. Ask a question!
query = "What are the key findings of this paper?"
result = qa({"query": query})

print("Question:", query)
print("Answer:", result["result"])
print("\nSource Documents:")
for doc in result["source_documents"]:
    print(doc.metadata["source"])
