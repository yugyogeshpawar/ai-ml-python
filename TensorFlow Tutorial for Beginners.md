# TensorFlow Tutorial for Beginners

Welcome to this tutorial on **TensorFlow**! TensorFlow is a powerful open-source library by Google that helps you build and train machine learning models. With TensorFlow, you can create models for tasks like image recognition, natural language processing, and more.

Let’s dive in step-by-step. By the end, you'll learn how to create and train a simple neural network using TensorFlow.

---

## Prerequisites

Before starting, ensure you have Python installed on your computer. You can verify this by running the following command in your terminal or command prompt:

```bash
python --version
```

If Python is not installed, [download and install it here](https://www.python.org/downloads/).

Additionally, you should have basic programming knowledge and an understanding of Python.

---

## Step 1: Install TensorFlow

Install TensorFlow using pip by running the following command:

```bash
pip install tensorflow
```

If you’re using a GPU for faster training, make sure to install the GPU version of TensorFlow. You can follow the [TensorFlow installation guide](https://www.tensorflow.org/install) for more details.

---

## Step 2: Create a Python File

1. Open your favorite code editor (e.g., VS Code, PyCharm, or Jupyter Notebook).
2. Create a new file and name it `tensorflow_intro.py`.

---

## Step 3: Build a Simple Neural Network

Here’s an example of how to create and train a neural network to classify numbers from the MNIST dataset (a collection of handwritten digits):

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to a range of 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into 1D array
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

---

## Step 4: Run the Code

1. Save the file.
2. Open your terminal or command prompt in the folder where the file is saved.
3. Run the script:

```bash
python tensorflow_intro.py
```

---

## What Happens Next?

1. TensorFlow will load the MNIST dataset and preprocess it.
2. It will train a neural network with your data over 5 epochs.
3. Finally, it will evaluate the trained model on test data and print the accuracy.

---

## Step 5: Customize the Model

### Change the Number of Epochs
Modify the number of epochs (how many times the model sees the training data) in the `model.fit()` function:

```python
model.fit(x_train, y_train, epochs=10)
```

Increasing epochs may improve accuracy but could lead to overfitting.

### Add More Layers
You can add more layers to make the model deeper:

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## What's Next?

Here are some fun things to explore:

1. **Use a Different Dataset**: Try using the CIFAR-10 dataset (a collection of 60,000 32x32 color images in 10 classes).
2. **Build a Convolutional Neural Network (CNN)**: CNNs are great for image-related tasks.
3. **Save and Load Models**: Learn how to save your trained model to use it later.

You can explore the [TensorFlow Documentation](https://www.tensorflow.org/) to learn more.

---

## Troubleshooting

- **Installation Issues**: Make sure you have installed the correct version of TensorFlow for your environment (CPU or GPU).
- **Slow Training**: If training is slow, consider using a GPU.
- **Dataset Errors**: Ensure the dataset is loaded and preprocessed correctly.

---

That’s it! You’ve just learned the basics of TensorFlow and created your first neural network. Keep experimenting and building amazing machine learning models!

