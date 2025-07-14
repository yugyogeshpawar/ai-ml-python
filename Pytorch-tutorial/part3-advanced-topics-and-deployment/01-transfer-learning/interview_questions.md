# Interview Questions: Transfer Learning

These questions assess your conceptual understanding of transfer learning strategies.

### 1. What is transfer learning and why is it so powerful for computer vision tasks?

**Answer:**
**Transfer learning** is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second, related task. In computer vision, this typically involves taking a large model pre-trained on a massive dataset like ImageNet and adapting it for a more specific task (e.g., classifying a particular type of flower).

It is powerful for several reasons:
1.  **Leverages Pre-existing Knowledge:** Models trained on ImageNet have already learned to recognize a rich hierarchy of visual features, from simple edges and textures to complex objects like eyes, wheels, and faces. This general knowledge is highly valuable and applicable to many other vision tasks.
2.  **Reduces Data Requirement:** Training a deep neural network from scratch requires a huge amount of labeled data. By using a pre-trained model, you can achieve high performance with a much smaller, task-specific dataset because the model isn't starting from zero.
3.  **Faster Convergence:** Since the model starts with well-initialized weights instead of random ones, it has a much better starting point. This leads to faster convergence and significantly reduces the required training time and computational cost.

In essence, transfer learning allows developers and researchers without access to massive datasets or computational resources to build highly effective, state-of-the-art models.

### 2. You are using a pre-trained ResNet-18 model for a new task. Describe the steps you would take to modify the model for "feature extraction". Which part of the model do you train?

**Answer:**
The "feature extraction" approach involves using the pre-trained model as a fixed feature extractor and only training a new, final classification layer.

The steps are as follows:
1.  **Load the pre-trained model:**
    ```python
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    ```
2.  **Freeze the pre-trained weights:** You must iterate through all the parameters of the model and set their `requires_grad` attribute to `False`. This "freezes" them, ensuring that they will not be updated by the optimizer during training.
    ```python
    for param in model.parameters():
        param.requires_grad = False
    ```
3.  **Replace the final layer:** The final layer of the pre-trained model (the "head" or "classifier") is designed for the original task (e.g., 1000 classes for ImageNet). You must replace it with a new layer that is suitable for your new task. For ResNet, this layer is called `fc`.
    ```python
    num_ftrs = model.fc.in_features
    num_classes_of_my_task = 10 # Example
    model.fc = nn.Linear(num_ftrs, num_classes_of_my_task)
    ```
    By default, this new layer's parameters will have `requires_grad=True`.

4.  **Train the model:** When you create an optimizer, you will pass it the model's parameters. The optimizer will only update the parameters of the layers where `requires_grad` is `True`. In this case, **only the weights of the new `model.fc` layer that you just added will be trained.**

### 3. What is the difference between the "feature extraction" and "fine-tuning" strategies in transfer learning?

**Answer:**
The key difference lies in **which parts of the pre-trained model are trained** on the new dataset.

-   **Feature Extraction:**
    -   **What is trained:** Only the parameters of the **newly added classification layer** are trained.
    -   **What is frozen:** The weights of all the other layers in the pre-trained model (the convolutional base) are **frozen** and are not updated.
    -   **When to use:** This is a good strategy when your new dataset is **small** and/or very similar to the original dataset (e.g., ImageNet). Since the pre-trained features are already very good, you just need to learn how to map those features to your new set of classes.

-   **Fine-Tuning:**
    -   **What is trained:** The parameters of the **entire model** are trained (or "fine-tuned"). You still replace the final classification layer, but you allow the optimizer to make small adjustments to the weights of the pre-trained layers as well.
    -   **What is frozen:** Nothing is frozen, but it's common practice to use a much **smaller learning rate** for the pre-trained layers than for the new classifier layer to prevent catastrophic forgetting of the learned features.
    -   **When to use:** This is a better strategy when your new dataset is **larger** and/or somewhat different from the original dataset. This allows the model to adapt its pre-existing feature knowledge more specifically to the nuances of your new data.
