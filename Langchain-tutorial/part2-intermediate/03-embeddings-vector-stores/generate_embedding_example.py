from langchain_openai import OpenAIEmbeddings

# Ensure your OPENAI_API_KEY is set as an environment variable.

# 1. Initialize the embedding model
embeddings_model = OpenAIEmbeddings()

# 2. Define a sentence you want to embed
sentence = "The quick brown fox jumps over the lazy dog."

# 3. Use the .embed_query() method to get the embedding vector
# This method is used for embedding single pieces of text (like a user query).
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
