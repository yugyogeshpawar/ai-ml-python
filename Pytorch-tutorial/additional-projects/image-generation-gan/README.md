# Additional Project: Image Generation with a GAN

This project explores a different side of deep learning: **generative models**. Instead of classifying existing data, we will build a model that can *generate* new data. Specifically, we will build a simple **Generative Adversarial Network (GAN)** to generate images of handwritten digits that look like they came from the MNIST dataset.

## What is a GAN?

A GAN consists of two neural networks that are trained simultaneously in a competitive, zero-sum game:

1.  **The Generator (`G`):** Its job is to create fake data that looks real. It takes a random noise vector as input and outputs a synthetic image. The Generator's goal is to fool the Discriminator into thinking its fake images are real.

2.  **The Discriminator (`D`):** Its job is to act as a detective. It takes an image (either a real one from the dataset or a fake one from the Generator) and tries to determine if it is real or fake. The Discriminator's goal is to get better at telling the difference.

### The Training Process

The two networks are trained in a constant back-and-forth:
-   The **Discriminator** is trained on a batch of real images (which it should classify as "real") and a batch of fake images from the Generator (which it should classify as "fake").
-   The **Generator** is trained based on the Discriminator's output. It gets rewarded (its loss decreases) if it can produce an image that the Discriminator incorrectly classifies as "real".

Over time, this adversarial process leads to both networks improving. The Generator gets better at creating realistic images, and the Discriminator gets better at telling them apart.

## Project Steps

The `gan_mnist.py` script implements this process:

1.  **Setup and Data Loading:**
    -   Define hyperparameters like learning rate, batch size, number of epochs, and the size of the latent (noise) vector.
    -   Load the MNIST dataset using `torchvision.datasets.MNIST` and create a `DataLoader`. We only need the images, not the labels.

2.  **Define the Models:**
    -   **`Discriminator`:** A simple feedforward neural network that takes a flattened 784-pixel image (28x28) and outputs a single logit. This logit represents the model's belief that the input image is real.
    -   **`Generator`:** Another feedforward network that takes a random noise vector (from the latent space) as input and outputs a 784-element vector, which can be reshaped into a 28x28 image.

3.  **Instantiate Models, Loss, and Optimizers:**
    -   Create instances of the Generator and Discriminator.
    -   The loss function is `nn.BCEWithLogitsLoss`, as the Discriminator is performing a binary classification task (real vs. fake).
    -   Create separate optimizers for the Generator and the Discriminator, as they are trained independently.

4.  **The Training Loop:**
    -   For each epoch, iterate through the `DataLoader`.
    -   **Train the Discriminator:**
        -   Take a batch of real images. Calculate the Discriminator's loss for these images (targets are all 1s for "real").
        -   Generate a batch of fake images using the Generator. Calculate the Discriminator's loss for these fake images (targets are all 0s for "fake").
        -   Add the real and fake losses, and perform a backward pass and optimizer step for the Discriminator only.
    -   **Train the Generator:**
        -   Generate a new batch of fake images.
        -   Pass them through the Discriminator.
        -   Calculate the Generator's loss. The key here is that the Generator wants the Discriminator to output "real", so we use a target of all 1s.
        -   Perform a backward pass and optimizer step for the Generator only.

5.  **Save and Visualize Results:**
    -   Periodically, the script saves a grid of images produced by the Generator, allowing you to visually track its progress as it learns to generate more convincing digits.

This project is a fascinating introduction to the world of generative AI and showcases a different kind of training dynamic than standard supervised learning.
