# Interview Questions: Saving and Loading Models

These questions cover the best practices for model persistence in PyTorch.

### 1. What is the recommended way to save a model in PyTorch, and why is it preferred over the alternative?

**Answer:**
The recommended way to save a model in PyTorch is to save only its **`state_dict`**.

This is done by calling `model.state_dict()` to get a dictionary of the model's parameters and then saving this dictionary with `torch.save()`.

This method is preferred over the alternative—saving the entire model object (`torch.save(model, ...)`)-for several key reasons:
1.  **Flexibility and Portability:** Saving the `state_dict` decouples the model's learned parameters from its Python class definition. This means you can load the weights into any model that has the same architecture, even if the surrounding code or file structure has changed. Saving the entire model pickles the whole object, creating a tight dependency on the exact file paths and class definitions used during saving, which is very brittle.
2.  **Longevity:** Code refactoring is common. If you rename a file or a directory, a model saved via the `state_dict` method can still be loaded, whereas a model where the entire object was saved will likely fail.
3.  **Lightweight:** The `state_dict` is just the parameters, making it the most essential and lightweight representation of the trained model.

Because of its robustness and flexibility, saving the `state_dict` is the standard and most professional approach.

### 2. You have loaded a trained model from a file to use it for making predictions on new data. What is the first thing you should do to the model after loading its `state_dict`, and why is this step critical?

**Answer:**
The first thing you must do is put the model into evaluation mode by calling **`model.eval()`**.

This step is critical because some layers in PyTorch behave differently during training and evaluation. The two most common examples are:
-   **Dropout Layers (`nn.Dropout`):** During training, dropout randomly sets a fraction of input units to zero at each update to prevent co-adaptation of neurons. During evaluation, you want to use the entire network to make a deterministic prediction, so `model.eval()` deactivates the dropout layers.
-   **Batch Normalization Layers (`nn.BatchNorm2d`):** During training, batch norm calculates the mean and standard deviation of the current batch of data to normalize it. During evaluation, it should use the fixed running statistics (mean and variance) that were learned and aggregated over the entire training process. `model.eval()` switches the layer to use these fixed statistics.

If you forget to call `model.eval()`, your model will still have these layers in training mode, leading to inconsistent, non-deterministic outputs and poor performance on the evaluation data.

### 3. What information would you typically save in a training "checkpoint" file, and why is it useful to save more than just the model's weights?

**Answer:**
A training checkpoint is a snapshot of the training process, designed to allow you to resume training seamlessly. While you could save just the model's weights, a comprehensive checkpoint typically includes:

1.  **The Model's `state_dict`:** This is the most crucial part, containing all the learned weights and biases.
2.  **The Optimizer's `state_dict`:** This is also very important. It saves the internal state of the optimizer, including its moving averages (in the case of Adam) and any other parameters it maintains. Without this, you would have to restart the optimization process from scratch, potentially losing the "momentum" the optimizer has built up and slowing down convergence.
3.  **The Current Epoch Number:** Saving the epoch number allows you to know exactly where you left off in the training schedule. This is essential for managing learning rate schedules and knowing how long the model has been trained.
4.  **The Last Recorded Loss:** Saving the last training or validation loss provides a quick reference for the model's performance at the time of saving.

Saving all this information in a single dictionary (`torch.save(checkpoint, ...)` ensures that when you load the checkpoint, you can restore the model, the optimizer, and your position in the training schedule, allowing you to resume training as if it had never been interrupted.
