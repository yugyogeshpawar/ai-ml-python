from langchain.document_loaders import WebBaseLoader

# Create a loader for a specific web page
# You can replace this URL with any other to load different content
loader = WebBaseLoader("https://www.zdnet.com/article/what-is-langchain-and-how-to-use-it-an-introduction/")

# Load the documents
# The .load() method fetches and parses the content, returning a list of Document objects.
documents = loader.load()

# Inspect the loaded documents
print(f"Loaded {len(documents)} document(s).")

for i, doc in enumerate(documents):
    print(f"\n--- Document {i+1} ---")
    print(f"Content (first 200 chars): {doc.page_content[:200]}")
    print(f"Metadata: {doc.metadata}")
