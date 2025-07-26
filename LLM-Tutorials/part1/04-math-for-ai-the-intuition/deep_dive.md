# Deep Dive: A Touch of Real Math

**Note:** This final optional section gives a very brief glimpse of what the actual math looks like. You are not expected to understand it in detail.

---

### 1. Vectors in Practice

In the lesson, we said a vector is a list of numbers. In Python code, this is often represented as an array or a list. A word embedding for a model like `word2vec` might have 300 dimensions.

So, the vector for "cat" wouldn't just be `[4.2, -1.5, 3.8]`, it would be a list of 300 numbers:
```
# A simplified representation of a 300-dimensional vector
vector_cat = [4.2, -1.5, 3.8, ..., 0.9] 
```

**Measuring Distance:**
How do we know that "cat" and "kitten" are close? We calculate the "distance" between their vectors. A common method is **Cosine Similarity**.

*   **Intuition:** It measures the angle between two vectors. If two vectors point in almost the same direction, the angle between them is very small (close to 0°), and their cosine similarity is close to 1. If they point in opposite directions, the angle is 180°, and the similarity is -1. If they are perpendicular (unrelated), the angle is 90°, and the similarity is 0.
*   **Result:** `cosine_similarity(vector_cat, vector_kitten)` would be a high value (e.g., 0.9), while `cosine_similarity(vector_cat, vector_car)` would be a very low value (e.g., 0.05).

---

### 2. Gradient Descent in Practice

The "walking downhill" analogy is powered by calculus, specifically **derivatives**.

**The Derivative: The Exact Slope**
A derivative is a tool from calculus that gives you the exact slope of a function at any given point. In our analogy, the derivative tells you precisely which way is downhill and how steep it is.

**The Loss Function:**
The "valley" is defined by a **loss function**, which calculates the error. A common one for regression is the **Mean Squared Error (MSE)**:

`MSE = (1/n) * Σ(y_true - y_pred)²`

*   `y_true` is the correct answer (e.g., the actual house price).
*   `y_pred` is the model's prediction.
*   We square the difference to make sure it's always positive and to penalize large errors more heavily.
*   We average this over all `n` data points.

**The Update Rule:**
The core of gradient descent is the update rule, which adjusts the model's weights.

`new_weight = old_weight - learning_rate * gradient`

*   **`old_weight`**: The current value of one of the millions of weights in the neural network.
*   **`gradient`**: This is the crucial part. It's the derivative of the loss function with respect to the `old_weight`. It tells us how much a small change in this specific weight will affect the total error of the model. A large gradient means this weight has a big impact on the error.
*   **`learning_rate`**: This is a small number (e.g., 0.001) that controls how big of a step we take downhill.
    *   If the learning rate is too large, we might overshoot the bottom of the valley entirely.
    *   If it's too small, it will take a very, very long time to get to the bottom.

The process of calculating the gradients for all the weights in a deep neural network is called **backpropagation**, which you saw in the first lesson's Deep Dive. It is the engine that makes training modern AI models possible.
