# Exercises: Deployment with TorchServe

These exercises are designed to familiarize you with the TorchServe command-line tools.

## Exercise 1: Archive a Different Model

**Task:** Use the `torch-model-archiver` to package a different pre-trained model from `torchvision`. Let's use `VGG16`.

1.  Write a short Python script to save the `state_dict` of a pre-trained `vgg16` model to a file named `vgg16.pth`.
2.  Find the model definition file for VGG from the official PyTorch vision repository and save it as `vgg.py`.
3.  Use the `torch-model-archiver` command to create a `vgg16.mar` file.
    -   Give it the model name `vgg16_classifier`.
    -   Use the `vgg.py` file as the `--model-file`.
    -   Use the `vgg16.pth` file as the `--serialized-file`.
    -   Use the same `index_to_name.json` file and `image_classifier` handler as the example.

**Goal:** Practice the model archiving step with a different architecture, which is the most critical part of the TorchServe workflow.

## Exercise 2: Use the Management API

**Task:** TorchServe has a Management API that runs on port 8081 by default. You can use it to register, unregister, and scale models without restarting the server.

1.  Start TorchServe with *no models loaded initially*.
    ```bash
    torchserve --start --ncs --model-store model_store --ts-config config.properties
    ```
2.  **Register the model:** Use `curl` to send a `POST` request to the Management API to register the `resnet18_classifier` model you created in the example.
    ```bash
    curl -X POST "http://127.0.0.1:8081/models?url=resnet18_classifier.mar&model_name=resnet18"
    ```
3.  **Check the status:** Use `curl` to see which models are currently loaded.
    ```bash
    curl "http://127.0.0.1:8081/models"
    ```
4.  **Unregister the model:** Use `curl` to unregister the model.
    ```bash
    curl -X DELETE "http://127.0.0.1:8081/models/resnet18"
    ```
5.  Check the status again to confirm it has been removed.

**Goal:** Understand how to manage models dynamically using the Management API, which is essential for production environments where you need to update models without downtime.

## Exercise 3: Create a Custom Handler

**Task:** The default `image_classifier` handler is great, but sometimes you need custom logic. Let's create a very simple custom handler.

1.  Create a file named `my_handler.py` with the following content. This handler inherits from the base handler but adds a custom message to the output.
    ```python
    from ts.torch_handler.base_handler import BaseHandler

    class MyCustomHandler(BaseHandler):
        def postprocess(self, data):
            # The default handler returns a list of dictionaries.
            # We'll just add a custom message to it.
            # In a real scenario, you could format the output differently.
            processed_data = super().postprocess(data)
            return [{"predictions": processed_data, "message": "Hello from my custom handler!"}]
    ```
2.  Re-archive your `resnet18` model, but this time, point the `--handler` flag to your custom script.
    ```bash
    torch-model-archiver ... --handler my_handler.py ...
    ```
3.  Start TorchServe and make a prediction request.
4.  Observe the new JSON output that includes your custom message.

**Goal:** Learn the basics of creating a custom handler to take full control over the pre-processing and post-processing steps of your model server.
