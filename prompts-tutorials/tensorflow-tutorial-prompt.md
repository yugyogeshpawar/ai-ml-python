# Prompt for Generating a Comprehensive TensorFlow Tutorial

## 1. Overall Objective
Generate a complete, multi-part TensorFlow tutorial suitable for hosting on GitHub. The tutorial should guide a user from a beginner level to an advanced level, covering the entire lifecycle of building, training, and deploying deep learning models with TensorFlow and Keras.

## 2. Target Audience
The tutorial is for Python developers, students, and researchers who want to learn deep learning using TensorFlow. A basic understanding of Python and machine learning concepts is helpful but not strictly required.

## 3. Core Philosophy & Style
- **Keras-First Approach:** Emphasize the high-level Keras API (`tf.keras`) for simplicity and ease of use, while introducing lower-level TensorFlow concepts where necessary.
- **Hands-On and Practical:** Every lesson must be grounded in practical code examples that are easy to run and understand.
- **Ecosystem-Aware:** Introduce key components of the TensorFlow ecosystem, such as TensorBoard and TensorFlow Serving.

## 4. High-Level Structure
The tutorial will be divided into three main parts, plus a section for projects.

- **Part 1: The Fundamentals of TensorFlow**
- **Part 2: Building and Training Neural Networks with Keras**
- **Part 3: Advanced Topics and Deployment**
- **Additional Projects**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following four files inside a dedicated topic directory (e.g., `part1-fundamentals/01-what-is-tensorflow/`):

1.  **`README.md`**: The main lesson file with detailed explanations.
2.  **`[topic]_example.py`**: A runnable Python script demonstrating the lesson's concept.
3.  **`exercises.md`**: A file containing 2-3 practical exercises.
4.  **`interview_questions.md`**: A file with 3 relevant interview questions and detailed answers.

---

### **Part 1: The Fundamentals of TensorFlow**
- **Goal:** Introduce the core data structures and mechanics of TensorFlow.
- **Topics:**
    1.  `01-what-is-tensorflow`: Introduction to TensorFlow, the role of Keras, and setting up the environment.
    2.  `02-tensorflow-tensors`: The `tf.Tensor` object, creating tensors, tensor operations, and NumPy interoperability.
    3.  `03-automatic-differentiation-and-gradienttape`: The concept of `tf.GradientTape` for automatic differentiation.
    4.  `04-variables-and-modules`: Understanding `tf.Variable` for trainable parameters and `tf.Module` for encapsulating model components.

- **Project:** A simple linear regression model from scratch using `tf.GradientTape` to train the weights.

### **Part 2: Building and Training Neural Networks with Keras**
- **Goal:** Cover the complete workflow for training a deep learning model using the `tf.keras` API.
- **Topics:**
    1.  `01-building-models-with-keras`: The `Sequential` and Functional APIs for defining models.
    2.  `02-common-layers`: Exploring core layers like `Dense`, `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dropout`.
    3.  `03-compiling-a-model`: Using `model.compile()` to configure the optimizer, loss function, and metrics.
    4.  `04-training-and-evaluation`: Using `model.fit()` for training and `model.evaluate()` for testing.
    5.  `05-saving-and-loading-models`: How to save and load entire models, weights, and configurations.

- **Project:** A complete image classification model using a Keras CNN on the Fashion MNIST dataset.

### **Part 3: Advanced Topics and Deployment**
- **Goal:** Explore more advanced workflows, architectures, and production-level deployment.
- **Topics:**
    1.  `01-the-tf-data-api`: Building efficient input pipelines with `tf.data.Dataset` for performance.
    2.  `02-transfer-learning-with-keras`: Using pre-trained models from `tf.keras.applications` (like MobileNetV2) for fine-tuning.
    3.  `03-visualizing-with-tensorboard`: Using the TensorBoard callback to visualize training metrics, model graphs, and more.
    4.  `04-deployment-with-tensorflow-serving`: An introduction to exporting a model to the SavedModel format and serving it with TensorFlow Serving.

- **Project:** A cat vs. dog image classification project using transfer learning and visualizing the results in TensorBoard.

### **Additional Projects**
- **Goal:** Provide more complex, real-world examples.
- **Projects:**
    1.  `text-generation-rnn`: A character-level Recurrent Neural Network (RNN) to generate text in the style of Shakespeare.
    2.  `time-series-forecasting`: A forecasting model to predict future values in a time-series dataset.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Explain concepts clearly, focusing on the intuitive Keras API.
    - Provide a "Step-by-Step Code Tutorial" section that walks through the example script.
- **For `[topic]_example.py` files:**
    - The code must be clean, runnable, and well-commented.
- **For `exercises.md` and `interview_questions.md` files:**
    - Ensure the content is practical and tests for a deep understanding of the TensorFlow/Keras workflow.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
