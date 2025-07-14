# Part 3, Topic 3: Deployment with TorchServe

Training a model is only half the battle. To be useful, a model needs to be **deployed** so that it can receive input data and return predictions in a real-world application. This process is called **inference**.

**TorchServe** is a flexible and easy-to-use tool for serving PyTorch models in production. It was developed by PyTorch and is the officially recommended way to deploy PyTorch models.

## Why Use a Serving Tool?

You could write your own web server (e.g., using Flask or FastAPI) to serve your model, but a dedicated tool like TorchServe provides many advantages out of the box:
-   **Production Ready:** It's a high-performance tool designed to handle multiple requests efficiently.
-   **Easy to Use:** It has a simple command-line interface for serving models.
-   **REST API:** It automatically creates a REST API endpoint for your model, making it easy for applications to interact with it.
-   **Model Management:** You can serve multiple models at once, load new versions of models without downtime, and monitor performance.
-   **Batching:** It can automatically batch incoming requests on the server-side to improve GPU utilization and throughput.

## The TorchServe Workflow

Deploying a model with TorchServe involves three main steps:

### Step 1: Install TorchServe

First, you need to install TorchServe and its dependencies.

```bash
# Install dependencies
pip install torchserve torch-model-archiver torch-workflow-archiver

# Install Java (TorchServe runs on a Java backend)
# On macOS:
brew install openjdk@17
# On Ubuntu:
sudo apt-get install openjdk-17-jdk
```

### Step 2: Package Your Model (`.mar` file)

To be served, your model needs to be packaged into a **Model Archive (`.mar`) file**. This file bundles together everything TorchServe needs to know about your model:
1.  **The serialized model weights (`.pth` file):** The `state_dict` you saved after training.
2.  **The model definition file (`.py` file):** The Python script containing your `nn.Module` class definition.
3.  **A handler script (`.py` file):** This is a crucial script that tells TorchServe how to handle requests. It defines:
    -   `preprocess()`: How to convert incoming request data (e.g., a JSON payload or an image file) into a tensor the model can understand.
    -   `inference()`: How to pass the tensor through the model.
    -   `postprocess()`: How to convert the model's output tensor back into a human-readable format (e.g., JSON).

You create the `.mar` file using the `torch-model-archiver` command-line tool.

**Example Command:**
```bash
torch-model-archiver --model-name my_image_classifier \
--version 1.0 \
--model-file model_definition.py \
--serialized-file model_weights.pth \
--handler image_classifier \
--export-path model_store
```
-   `--handler`: TorchServe provides default handlers for common tasks like `image_classifier` and `text_classifier`, so you often don't need to write your own from scratch.
-   `--export-path`: This specifies the directory where the `.mar` file will be saved. This directory is called the **model store**.

### Step 3: Start TorchServe and Register the Model

Once you have your `.mar` file in a model store directory, you can start the TorchServe server.

**Start the Server:**
```bash
# This starts the server and tells it where to look for models.
torchserve --start --model-store model_store --models my_image_classifier.mar
```
The `--models` argument tells TorchServe to load and serve the `my_image_classifier` model immediately.

**Make a Prediction:**
Once the server is running, it exposes several API endpoints. The most important one is the `predictions` endpoint. You can send a request to it using a tool like `curl`.

```bash
# Send an image file to the model for prediction
curl http://127.0.0.1:8080/predictions/my_image_classifier -T kitten.jpg
```

The server will respond with a JSON object containing the model's prediction.

## Summary

TorchServe provides a standardized and powerful workflow for deploying PyTorch models. By packaging your model and its logic into a `.mar` file, you create a self-contained, portable artifact that can be easily served in any environment, from local development to a large-scale cloud deployment.

The `deployment_with_torchserve_example.py` file in this lesson will not be a runnable script itself, but rather a guide containing the necessary code snippets and shell commands to walk you through this entire process.
