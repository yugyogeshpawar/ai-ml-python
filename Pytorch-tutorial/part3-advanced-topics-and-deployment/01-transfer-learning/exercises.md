# Exercises: Transfer Learning

These exercises will help you practice using pre-trained models.

## Exercise 1: Use a Different Pre-trained Model

**Task:** The `torchvision.models` library contains many different architectures. Modify the `transfer_learning_example.py` script to use a different pre-trained model, such as `VGG16`.

1.  Load `models.vgg16(pretrained=True)`.
2.  **Investigate the model structure:** Print the `vgg16` model object. Unlike ResNet, which has a final layer named `fc`, VGG has a `classifier` section which is an `nn.Sequential` module. You will need to replace the *last* layer inside this classifier.
3.  Freeze the parameters of the "features" section of the model.
4.  Replace the last layer of the `vgg16.classifier` with a new `nn.Linear` layer appropriate for the number of classes in the hymenoptera dataset.
5.  Update the optimizer to pass the new layer's parameters to it.
6.  Run the training script.

**Goal:** Understand that different pre-trained models have different architectures and that you need to inspect the model to find the correct final layer to replace.

## Exercise 2: Filter Parameters for the Optimizer

**Task:** In the example, we created a list of parameters to update using a list comprehension: `params_to_update = [p for p in model.parameters() if p.requires_grad]`.

Write a short script that does the same thing but uses a standard `for` loop instead.

1.  Load a pre-trained model (e.g., `resnet18`).
2.  Freeze its parameters and replace the final layer as in the example.
3.  Create an empty list, `params_to_update`.
4.  Loop through `model.named_parameters()`.
5.  Inside the loop, use an `if` statement to check if `param.requires_grad` is `True`.
6.  If it is, append the parameter to your list and print its name.
7.  Finally, create an optimizer using this list of parameters.

**Goal:** Reinforce your understanding of how to programmatically select which parts of a model you want to train.

## Exercise 3: Fine-Tuning vs. Feature Extraction

**Task:** Briefly answer the following questions in your own words.

1.  What is the main difference between "fine-tuning" and "feature extraction" in the context of transfer learning?
2.  Under what circumstances would you choose feature extraction over fine-tuning? (Hint: think about dataset size and similarity to ImageNet).
3.  When fine-tuning, why is it common to use a smaller learning rate for the pre-trained convolutional layers compared to the newly added classifier layer?

**Goal:** Solidify your conceptual understanding of the two main transfer learning strategies and when to apply them.
