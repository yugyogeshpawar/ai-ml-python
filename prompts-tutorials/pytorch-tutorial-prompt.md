# Prompt for Generating a Comprehensive PyTorch Tutorial

## 1. Overall Objective
Generate a complete, multi-part PyTorch tutorial suitable for hosting on GitHub. The tutorial should guide a user from a beginner level to an advanced level, covering the entire lifecycle of building, training, and deploying deep learning models with PyTorch.

## 2. Target Audience
The tutorial is for Python developers, students, and researchers who want to learn deep learning using PyTorch. A basic understanding of Python and machine learning concepts is helpful but not strictly required.

## 3. Core Philosophy & Style
- **Conceptual Clarity:** Explain core deep learning and PyTorch concepts with simple analogies and clear, concise language.
- **Hands-On and Practical:** Every lesson must be grounded in practical code examples that are easy to run and understand.
- **Progressive Learning:** The tutorial should build on itself, with each part providing the foundation for the next.

## 4. High-Level Structure
The tutorial will be divided into three main parts, plus a section for projects.

- **Part 1: The Fundamentals of PyTorch**
- **Part 2: Building and Training Neural Networks**
- **Part 3: Advanced Topics and Deployment**
- **Additional Projects**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following four files inside a dedicated topic directory (e.g., `part1-fundamentals/01-what-is-pytorch/`):

1.  **`README.md`**: The main lesson file with detailed explanations.
2.  **`[topic]_example.py`**: A runnable Python script demonstrating the lesson's concept.
3.  **`exercises.md`**: A file containing 2-3 practical exercises.
4.  **`interview_questions.md`**: A file with 3 relevant interview questions and detailed answers.

---

### **Part 1: The Fundamentals of PyTorch**
- **Goal:** Introduce the core data structures and mechanics of PyTorch.
- **Topics:**
    1.  `01-what-is-pytorch`: Introduction to PyTorch, its advantages, and setting up the environment.
    2.  `02-pytorch-tensors`: The `torch.Tensor` object, creating tensors, tensor operations, and NumPy interoperability.
    3.  `03-autograd-automatic-differentiation`: The concept of the computation graph, how `autograd` works, and calculating gradients.
    4.  `04-building-a-basic-neural-network`: Introduction to `torch.nn`, defining a simple network with `nn.Module`, and understanding layers.

- **Project:** A simple linear regression model from scratch to predict a value based on a single input.

### **Part 2: Building and Training Neural Networks**
- **Goal:** Cover the complete workflow for training a deep learning model on a real dataset.
- **Topics:**
    1.  `01-datasets-and-dataloaders`: The `Dataset` and `DataLoader` classes for loading and batching data efficiently.
    2.  `02-transforms`: Pre-processing and augmenting data with `torchvision.transforms`.
    3.  `03-loss-functions`: Understanding different loss functions (`nn.CrossEntropyLoss`, `nn.MSELoss`) and when to use them.
    4.  `04-optimizers`: The role of optimizers, exploring `torch.optim` (e.g., Adam, SGD), and the training loop.
    5.  `05-saving-and-loading-models`: How to save and load model weights (`state_dict`) for inference or continued training.

- **Project:** A complete image classification model using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

### **Part 3: Advanced Topics and Deployment**
- **Goal:** Explore more advanced architectures, techniques, and production-level deployment.
- **Topics:**
    1.  `01-transfer-learning`: Using pre-trained models (like ResNet) from `torchvision.models` for fine-tuning on a custom dataset.
    2.  `02-recurrent-neural-networks-rnn`: Building an RNN for sequence data, such as text classification.
    3.  `03-deployment-with-torchserve`: An introduction to serving PyTorch models as a production-ready API using TorchServe.
    4.  `04-gpu-and-cuda`: How to move tensors and models to the GPU for accelerated training (`.to(device)`).

- **Project:** A sentiment analysis project using a pre-trained language model and deploying it as a local API.

### **Additional Projects**
- **Goal:** Provide more complex, real-world examples.
- **Projects:**
    1.  `image-generation-gan`: A simple Generative Adversarial Network (GAN) to generate images of handwritten digits (MNIST).
    2.  `object-detection-yolo`: An introduction to using a pre-trained object detection model like YOLO to find objects in images.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Explain concepts clearly with diagrams where helpful.
    - Provide a "Step-by-Step Code Tutorial" section that walks through the example script.
- **For `[topic]_example.py` files:**
    - The code must be clean, runnable, and well-commented.
- **For `exercises.md` and `interview_questions.md` files:**
    - Ensure the content is practical and tests for a deep understanding of the topic.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
