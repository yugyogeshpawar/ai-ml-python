# Try It Yourself: Developing Your Mathematical Intuition

You can build an intuition for these concepts without ever touching a formula. These exercises use web-based tools and your own imagination.

---

### Exercise 1: Exploring the "Map of Meaning" (Vectors)

This exercise lets you see a real-world "embedding space" or "map of meaning" that a model has learned.

1.  **Go to the "Embedding Projector" by TensorFlow.** This is a powerful tool for visualizing vectors.
    *   **Click here to open the tool:** [https://projector.tensorflow.org/](https://projector.tensorflow.org/)
    *   It will load a 3D map of thousands of words. Each dot is a word, and its position is determined by its vector.

2.  **Explore the map.** Use your mouse to rotate, pan, and zoom. You are flying through a model's "brain."

3.  **Search for a word.** In the search bar on the right-hand side, type a simple word like **"dog"** and press Enter.
    *   The projector will highlight the point for "dog."
    *   Look at the list of "Nearest points" that appears on the right. You will see words like "dogs," "puppy," "pet," and "cat." These are the closest neighbors to "dog" on the map.

4.  **Try a few other words.** Search for the following and look at their neighbors:
    *   **"Happy":** You'll find words like "glad," "sad," "pleased."
    *   **"Car":** You'll find "cars," "vehicle," "truck," "road."
    *   **"King":** You'll find "queen," "prince," "monarch," "throne."

5.  **Reflect:** You are seeing visual proof that the model has learned the relationships between words. It didn't just memorize a dictionary; it built a map where the distances between points represent the relationships between their meanings.

---

### Exercise 2: The "Human Gradient Descent" Game

This is a thought experiment to help you feel the intuition behind gradient descent.

1.  **Imagine you are building the perfect paper airplane.** Your goal is to make it fly as far as possible. The "error" or "loss" is how short of the world record your flight is. A shorter flight means a higher error.

2.  **You start with a random design.** You fold it based on a guess and throw it. It flies 5 feet. The error is huge.

3.  **Now, you apply gradient descent.** You don't just randomly fold a new plane. You think: "Okay, what one small change can I make to improve the flight?"
    *   **You decide to change one thing:** the angle of the wings. You make a tiny adjustment—folding them up *slightly* more than before.
    *   You throw the new plane. It flies 8 feet! **The error has decreased.** You have successfully taken a "step" in the right "downhill" direction.

4.  **You repeat the process.** From your new design, you again ask: "What's the *next* small change I can make?"
    *   Maybe you try making the nose slightly heavier. You add a paperclip.
    *   You throw it. It flies 15 feet! Another successful step.

5.  **You hit a problem.** You try making the nose even heavier by adding a second paperclip.
    *   You throw it. It nosedives and only flies 3 feet. **The error has increased.** You took a step in the wrong, "uphill" direction.
    *   So, you undo that change and go back to the one-paperclip design.

6.  **Reflect:** This is how an AI learns. It's a slow, iterative process of making tiny adjustments to its internal parameters (the "folds" of the airplane) and keeping the changes that reduce the error, while discarding the ones that increase it. After millions of these tiny adjustments, it arrives at a design that is highly optimized for its goal.
