# Exercises: Loss Functions

These exercises will help you practice selecting and using the correct loss function for different tasks.

## Exercise 1: Calculate MSE Loss Manually

**Task:** Given the predicted values `[1.0, 2.0, 3.0]` and the true values `[1.5, 2.5, 3.5]`:
1.  Calculate the Mean Squared Error (MSE) loss by hand.
    -   Find the difference for each pair.
    -   Square each difference.
    -   Find the average of the squared differences.
2.  Write a PyTorch script using `nn.MSELoss` to verify your manual calculation.

**Goal:** Solidify your understanding of what the MSE loss value actually represents.

## Exercise 2: Match Loss Functions to Problems

**Task:** For each of the following scenarios, identify the most appropriate loss function from this list: `nn.MSELoss`, `nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss`.

1.  **Scenario A:** Predicting the age of a person from a photograph.
2.  **Scenario B:** Classifying a news article into one of five categories: "Sports", "Politics", "Technology", "Entertainment", or "Business".
3.  **Scenario C:** Determining whether a customer will click on an online ad (Yes/No).
4.  **Scenario D:** A "tagging" system where a photo can be tagged with multiple labels like "beach", "sunset", and "vacation" simultaneously. (Hint: This is a multi-label classification problem. Think about how you could frame it as multiple binary decisions).

**Goal:** Practice the critical skill of selecting the right loss function based on the problem definition.

## Exercise 3: Cross-Entropy-Loss Input Shapes

**Task:** You have a batch of 16 images, and your task is to classify each image into one of 10 classes. Your model has just produced its output logits.

1.  What should be the shape of your model's output tensor (the predictions)?
2.  What should be the shape of your target tensor (the true labels)?
3.  Create two random tensors with these correct shapes and pass them to `nn.CrossEntropyLoss` to ensure they work without errors.

**Goal:** Reinforce your understanding of the specific input shapes required by `nn.CrossEntropyLoss`, which is a common source of errors for beginners.
