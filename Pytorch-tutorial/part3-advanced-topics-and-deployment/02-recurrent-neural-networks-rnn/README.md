# Part 3, Topic 2: Recurrent Neural Networks (RNNs)

So far, we have worked with feedforward neural networks (including CNNs), where the data flows in one direction from input to output. These networks are powerful, but they have a major limitation: they have no memory of past inputs. This makes them unsuitable for tasks involving **sequential data**, where the order of information is important.

Examples of sequential data include:
-   Text (a sequence of words or characters)
-   Time-series data (e.g., stock prices over time)
-   Audio and video

**Recurrent Neural Networks (RNNs)** are a class of neural networks designed specifically to handle sequential data.

## How do RNNs Work?

The key idea behind an RNN is the **hidden state**. An RNN processes a sequence one element at a time. At each step, the RNN takes two inputs:
1.  The current element of the sequence (e.g., a word in a sentence).
2.  The **hidden state** from the previous step.

The hidden state acts as the network's "memory," carrying information from past elements of the sequence. The RNN combines the current input with the previous hidden state to produce an output for the current step and a **new hidden state** to be passed to the next step.

This recurrent, looping mechanism allows the network to maintain a memory of the sequence it has seen so far.

## RNN Layers in PyTorch

PyTorch provides several powerful, pre-built RNN layers in the `torch.nn` module.

### `nn.RNN`

This is the most basic RNN layer. While simple, it often suffers from the **vanishing gradient problem**, making it difficult to learn long-range dependencies in a sequence.

### `nn.LSTM` (Long Short-Term Memory)

LSTMs are a more advanced and popular type of RNN designed to solve the vanishing gradient problem. They introduce a more complex internal structure with three "gates":
-   **Forget Gate:** Decides what information to throw away from the cell state.
-   **Input Gate:** Decides which new information to store in the cell state.
-   **Output Gate:** Decides what to output based on the cell state.

This gating mechanism allows the LSTM to selectively remember or forget information over long sequences, making it much more powerful than a simple RNN.

### `nn.GRU` (Gated Recurrent Unit)

GRUs are a variation of LSTMs that are slightly simpler, with fewer parameters. They combine the forget and input gates into a single "update gate." GRUs often perform similarly to LSTMs but are computationally a bit more efficient.

## Using an LSTM Layer

Let's look at the key inputs and outputs of an `nn.LSTM` layer.

**Initialization:**
```python
# input_size: The number of expected features in the input x
# hidden_size: The number of features in the hidden state h
# num_layers: Number of recurrent layers to stack.
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
```
-   `batch_first=True` is a very important argument. It means the input and output tensors will have the batch dimension as the first dimension (`batch_size, sequence_length, input_size`), which is more intuitive.

**Inputs to the LSTM:**
1.  **`input`**: A tensor of shape `(batch_size, sequence_length, input_size)`.
2.  **`(h_0, c_0)`** (optional): A tuple containing the initial hidden state and initial cell state. If not provided, they default to zeros.

**Outputs from the LSTM:**
1.  **`output`**: A tensor of shape `(batch_size, sequence_length, hidden_size)` containing the output features from the last LSTM layer for each time step.
2.  **`(h_n, c_n)`**: A tuple containing the final hidden state and final cell state for the entire sequence.

## Building an RNN for Text Classification

A common use case for an RNN is text classification (e.g., sentiment analysis). Here's a high-level overview of how you would build such a model:

1.  **Embedding Layer (`nn.Embedding`):** Text is first converted into numerical indices. An embedding layer then maps each index to a dense vector representation (a word embedding). This layer is learnable.
2.  **LSTM Layer (`nn.LSTM`):** The sequence of word embeddings is fed into an LSTM layer. The LSTM processes the sequence and captures contextual information.
3.  **Final Classifier (`nn.Linear`):** You typically take the output of the LSTM from the **last time step** (which represents a summary of the entire sentence) and pass it to a fully connected layer to produce the final class scores.

The `rnn_example.py` script provides a complete example of building an LSTM-based model for a simple text classification task.
