# deployment_with_torchserve_example.py
# This file is a step-by-step guide, not a single runnable script.
# It contains the code and commands needed to deploy a pre-trained
# image classification model with TorchServe.

# --- Objective ---
# We will deploy a pre-trained ResNet-18 model that can classify images
# from the ImageNet dataset.

# --- Step 1: Prepare the Model and Required Files ---

# We need three things:
# 1. The model's weights (.pth file)
# 2. The model's class definition (.py file)
# 3. A mapping file for the class labels (index_to_name.json)

# Let's create them.

# a) Save the pre-trained model weights
# =======================================
# Create a file named `save_model_weights.py` and run it.
"""
# save_model_weights.py
import torchvision.models as models
import torch

# Load a pre-trained ResNet-18
model = models.resnet18(pretrained=True)

# Save the state_dict
torch.save(model.state_dict(), 'resnet18.pth')
print("Saved ResNet-18 weights to resnet18.pth")
"""

# b) Get the model definition file
# =================================
# TorchServe needs the Python file containing the model's class definition.
# For standard torchvision models, you don't need to write this yourself.
# You can find the official ResNet implementation here:
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# Download this file and save it as `resnet.py`.
# For this tutorial, we assume you have `resnet.py` in your directory.
# (In a real project, you would use your own model's definition file).

# c) Create the class mapping file
# =================================
# We need a JSON file that maps the output indices of the model to human-readable class names.
# Create a file named `index_to_name.json`. You can find the full ImageNet list online.
# Here is a shortened example:
"""
{
    "0": "tench, Tinca tinca",
    "1": "goldfish, Carassius auratus",
    "2": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    ...
    "999": "toilet tissue, toilet paper, bathroom tissue"
}
"""
# For this example, we'll use a dummy version.
# Create `index_to_name.json` with the content: `{"0": "cat", "1": "dog"}`
# (This is just for the archiver to work; the real ResNet-18 is trained on 1000 classes).


# --- Step 2: Archive the Model using `torch-model-archiver` ---

# Now, we package everything into a single `.mar` file.
# Open your terminal in the directory containing the files from Step 1.

# Command to run in the terminal:
"""
torch-model-archiver --model-name resnet18_classifier \
--version 1.0 \
--model-file resnet.py \
--serialized-file resnet18.pth \
--extra-files index_to_name.json \
--handler image_classifier \
--export-path model_store
"""

# Breakdown of the command:
# --model-name: The name we'll use to call our model's API endpoint.
# --version: The model version.
# --model-file: The Python file with the model class (our downloaded resnet.py).
# --serialized-file: The saved model weights.
# --extra-files: Any other files the handler needs, like our class mapping.
# --handler: Specifies the logic for pre/post-processing. `image_classifier` is a built-in handler.
# --export-path: The directory to save the output `.mar` file. This will be our "model store".

# After running this, you should have a `model_store` directory with
# `resnet18_classifier.mar` inside it.


# --- Step 3: Start TorchServe and Serve the Model ---

# First, create a `config.properties` file to tell TorchServe which port to use
# and what the default models are.
# Create a file named `config.properties` with this content:
"""
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
"""

# Now, start the TorchServe server from your terminal.
# Command to run in the terminal:
"""
torchserve --start --ncs --model-store model_store --models resnet18_classifier.mar --ts-config config.properties
"""

# --start: Starts the server.
# --ncs: No config snapshot, to avoid issues on restart.
# --model-store: Points to our model store directory.
# --models: A shortcut to load and register `resnet18_classifier.mar` on startup.
# --ts-config: Points to our configuration file.

# If successful, the server is now running and ready for requests.


# --- Step 4: Make a Prediction ---

# Download a sample image of a cat and save it as `cat.jpg`.
# You can find one easily online.

# Send a prediction request using `curl`.
# Command to run in the terminal:
"""
curl http://127.0.0.1:8080/predictions/resnet18_classifier -T cat.jpg
"""

# Expected Output:
# You should receive a JSON response from the server with the predicted class.
# Because our `index_to_name.json` is incomplete, it might not show the correct
# class name, but it will show the predicted class index. With a full
# ImageNet index file, it would correctly identify the cat breed.
# Example response:
# {
#   "tabby, tabby cat": 0.4324,
#   "tiger cat": 0.321,
#   ...
# }


# --- Step 5: Stop the Server ---

# When you are finished, you can stop TorchServe.
# Command to run in the terminal:
"""
torchserve --stop
"""
