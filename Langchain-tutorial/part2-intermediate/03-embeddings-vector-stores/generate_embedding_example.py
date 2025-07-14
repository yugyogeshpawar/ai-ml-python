from langchain.embeddings import HuggingFaceEmbeddings

# 1. Initialize the local embedding model
# This uses a model from Hugging Face that runs on your machine.
# The first time you run this, it will download the model (approx. 90MB).
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Define a sentence you want to embed
sentence = "The quick brown fox jumps over the lazy dog."

# 3. Use the .embed_query() method to get the embedding vector
embedding_vector = embeddings_model.embed_query(sentence)

# 4. Print the results
print(f"Sentence: '{sentence}'")
print(f"\nEmbedding Vector (first 10 dimensions): {embedding_vector[:10]}")
print(f"\nTotal dimensions: {len(embedding_vector)}")

# You can also embed multiple documents at once using .embed_documents()
list_of_sentences = [
    "The cat sat on the mat.",
    "A feline enjoys a nap.",
    "The dog chased the ball."
]

list_of_vectors = embeddings_model.embed_documents(list_of_sentences)

print("\n--- Embedding multiple documents ---")
for sentence, vector in zip(list_of_sentences, list_of_vectors):
    print(f"\nSentence: '{sentence}'")
    print(f"Embedding Vector (first 10 dimensions): {vector[:10]}")
