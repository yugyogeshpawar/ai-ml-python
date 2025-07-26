# Examples in Action: Embeddings

Embeddings translate concepts into a "map of meaning." Here are concrete examples of how this map is used to power intelligent applications.

### 1. Semantic Search
This is the most direct application of embeddings. It's about searching for meaning, not just keywords.

*   **The Goal:** You want to build a search engine for your company's internal documents. An employee should be able to ask a question in natural language and find the relevant policy document.
*   **The Process:**
    1.  **Indexing:** You take every document (or paragraph) in your company's knowledge base, and you use an embedding model (like `text-embedding-3-small` from OpenAI) to calculate the embedding vector for each one. You store these vectors in a vector database.
    2.  **Querying:** An employee asks, "How many days of paid time off do I get per year?"
    3.  **Semantic Search:** Your system takes the employee's question and calculates its embedding vector. It then queries the vector database to find the document chunks whose vectors are most similar (closest on the "map") to the question's vector.
*   **The Result:** The search will return the paragraph from the HR policy manual that says, "Annual leave for full-time employees is allocated at 20 days per annum," even though this sentence doesn't contain the words "paid time off." The system works because the *meaning* of "paid time off" and "annual leave" is represented by very similar vectors.

---

### 2. Clustering and Topic Modeling
Embeddings can be used to automatically group similar items together without any pre-existing labels (an unsupervised learning task).

*   **The Goal:** You have 10,000 customer support tickets, and you want to understand the main reasons customers are contacting you.
*   **The Process:**
    1.  **Embedding:** You calculate the embedding vector for each support ticket.
    2.  **Clustering:** You use a clustering algorithm (like K-Means) to group these vectors into, for example, 10 distinct clusters in the embedding space.
*   **The Result:** You can then examine the tickets within each cluster to see what they have in common. You might find:
    *   **Cluster 1:** Full of tickets with words like "password," "reset," "forgot," "locked out." (This is the "Account Access" cluster).
    *   **Cluster 2:** Full of tickets with words like "late," "shipping," "where is," "tracking." (This is the "Shipping Issues" cluster).
    *   **Cluster 3:** Full of tickets with words like "broken," "doesn't work," "defective." (This is the "Product Faults" cluster).
    You have automatically discovered the main topics of your customer complaints without having to read and categorize them manually.

---

### 3. Anomaly Detection
By understanding what is "normal," embeddings can help identify things that are unusual or out of place.

*   **The Goal:** You want to detect fraudulent credit card transactions.
*   **The Process:**
    1.  For a given user, you can create an embedding vector that represents their "normal" spending behavior. This might be an average of the embeddings of all their typical transactions (e.g., "groceries in San Francisco," "coffee shop," "gas station"). This creates a "normal" point for them on the map of meaning.
    2.  When a new transaction comes in (e.g., "jewelry store in another country"), you calculate its embedding vector.
    3.  You then measure the distance between the new transaction's vector and the user's "normal" vector.
*   **The Result:** If the distance is very large, it means the new transaction is semantically very different from the user's usual behavior. This signals an anomaly, and the system can flag the transaction for review.
