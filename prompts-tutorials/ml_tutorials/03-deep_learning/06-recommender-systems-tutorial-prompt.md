# Prompt: Building a Deep Learning Recommender System

### 1. Title
Generate a tutorial titled: **"Building a Movie Recommender: An Introduction to Deep Learning for Recommender Systems"**

### 2. Objective
To provide a hands-on guide to building a modern recommender system using deep learning. The reader will learn the principles of collaborative filtering and implement a matrix factorization model in PyTorch to predict movie ratings.

### 3. Target Audience
*   Aspiring ML engineers and data scientists interested in personalization and recommendation.
*   Developers who want to understand the technology behind platforms like Netflix and Spotify.
*   Students looking for a practical, real-world deep learning project.

### 4. Prerequisites
*   Strong Python programming skills.
*   Solid experience with PyTorch, including building custom `nn.Module` classes and writing training loops.
*   Familiarity with `pandas` for data manipulation.

### 5. Key Concepts Covered
*   **Collaborative Filtering:** The core idea of recommending items based on the behavior of similar users.
*   **Matrix Factorization:** A powerful technique for learning latent features (embeddings) for users and items.
*   **Embeddings:** Representing users and items as dense vectors in a shared latent space.
*   **The Dot Product:** How the similarity between a user's and an item's embedding can predict a rating.
*   **Implicit vs. Explicit Feedback:** The difference between ratings (explicit) and clicks/views (implicit).

### 6. Open-Source Tools & Libraries
*   **Python 3.x**
*   **PyTorch:** For building and training the deep learning model.
*   **pandas:** For loading and processing the data.
*   **scikit-learn:** For splitting the data.

### 7. Dataset
*   The **MovieLens 100k Dataset**. It's a classic, publicly available dataset containing 100,000 ratings from 1,000 users on 1,700 movies.

### 8. Step-by-Step Tutorial Structure

**Part 1: The Art of Recommendation**
*   **1.1 How Netflix Knows What You'll Like:** Introduce the concept of collaborative filtering with an intuitive example: "If User A and User B both liked the same 5 movies, and User A also liked a 6th movie, we should probably recommend that 6th movie to User B."
*   **1.2 Matrix Factorization with Embeddings:**
    *   Explain the goal: to predict the missing values in a huge user-item rating matrix.
    *   Introduce matrix factorization as a way to solve this. Use a diagram to show how the large, sparse rating matrix can be approximated by the product of two smaller, dense matrices: a user-embedding matrix and an item-embedding matrix.

**Part 2: Preparing the MovieLens Data**
*   **2.1 Goal:** Load the data and prepare it for training.
*   **2.2 Implementation:**
    1.  Download and load the MovieLens 100k dataset using `pandas`.
    2.  Map the raw user and movie IDs to continuous integer indices.
    3.  Split the data into training and validation sets.

**Part 3: Building the Matrix Factorization Model in PyTorch**
*   **3.1 Goal:** Create a PyTorch `nn.Module` that implements the matrix factorization model.
*   **3.2 Implementation:**
    1.  Create a class that inherits from `nn.Module`.
    2.  In the `__init__` method, define two `nn.Embedding` layers: one for users and one for movies. Explain that these layers are essentially just lookup tables for the embedding vectors.
    3.  In the `forward` method, which takes user and movie indices as input:
        *   Look up the embedding vectors for the given users and movies.
        *   Calculate the dot product of the user and movie embeddings to get the predicted rating.

**Part 4: Training the Recommender**
*   **4.1 Goal:** Write a PyTorch training loop to learn the optimal embeddings.
*   **4.2 Implementation:**
    1.  Instantiate the model, optimizer (Adam), and loss function (**MSELoss**, since this is a regression problem).
    2.  Create a `DataLoader` to batch the training data.
    3.  Write a standard training loop that iterates through the data, calculates the loss between the predicted ratings and the actual ratings, and updates the model's weights (the embeddings).
    4.  Monitor the validation loss to check for overfitting.

**Part 5: Making Movie Recommendations**
*   **5.1 Goal:** Use the trained model to generate personalized movie recommendations for a specific user.
*   **5.2 Implementation:**
    1.  Take a user ID as input.
    2.  Get the user's embedding vector from the trained model.
    3.  Calculate the dot product of this user's embedding with the embedding of *every single movie* in the dataset. This gives a predicted rating for all movies.
    4.  Sort the movies by their predicted rating in descending order.
    5.  Return the top N movies that the user has not already rated.

**Part 6: Conclusion**
*   Recap the process of building a deep learning-based recommender system.
*   Emphasize that the model learned the "taste" of each user and the "genre profile" of each movie automatically, just by looking at ratings.
*   Discuss next steps, such as content-based filtering and building hybrid recommender systems.

### 9. Tone and Style
*   **Tone:** Practical, insightful, and application-focused.
*   **Style:** Connect the abstract deep learning concepts (like embeddings) to the concrete, real-world goal of making good recommendations. The code should be a clean, standard PyTorch implementation.
