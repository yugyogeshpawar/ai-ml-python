import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI
from langchain import hub

# Ensure your OPENAI_API_KEY is set as an environment variable

# Create a dummy PDF file for demonstration
with open("your_document.pdf", "w") as f:
    f.write("This is a dummy PDF.")

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
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    OpenAI(), retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


# 6. Ask a question!
query = "What are the key findings of this paper?"
result = retrieval_chain.invoke({"input": query})

print("Question:", query)
print("Answer:", result["answer"])
print("\nSource Documents:")
for doc in result["context"]:
    print(doc.metadata["source"])
