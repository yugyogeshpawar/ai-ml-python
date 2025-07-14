# Interview Questions: Recurrent Neural Networks

These questions test your understanding of the architecture and challenges of RNNs.

### 1. What is the "vanishing gradient problem" in the context of simple RNNs, and how do LSTMs and GRUs address it?

**Answer:**
The **vanishing gradient problem** occurs during the training of deep networks, and particularly in simple RNNs when processing long sequences. During backpropagation, gradients are calculated by repeatedly applying the chain rule. In a simple RNN, this involves multiplying by the same weight matrix at each time step. If the values in this weight matrix are small (less than 1), the gradients can shrink exponentially as they are propagated back through time.

This means that for long sequences, the gradients for the earlier time steps can become incredibly small ("vanish"), effectively becoming zero. As a result, the model cannot learn the influence of these earlier time steps on the final output, making it impossible to capture long-range dependencies.

**How LSTMs and GRUs address it:**
LSTMs and GRUs were specifically designed to combat this problem using a **gating mechanism**.
-   **LSTMs** introduce a separate **cell state** that acts as a conveyor belt, allowing information to flow through the network largely unchanged. The flow of information is controlled by three gates (input, forget, and output gates). The **forget gate** can choose to keep information from previous states, and the **input gate** can add new information, all without necessarily shrinking the gradient at each step. This allows gradients to flow more easily over many time steps.
-   **GRUs** have a simpler architecture with an "update gate" that combines the forget and input gates. It also effectively controls how much information from the previous state is carried forward, preventing the gradient from vanishing.

### 2. In an LSTM, what is the difference between the "hidden state" and the "cell state"?

**Answer:**
While both are part of the LSTM's "memory," they have different roles:

-   **Cell State (`c_t`):** This is often considered the **long-term memory** of the LSTM. It runs straight down the entire sequence, with only minor linear interactions. The gating mechanism can add information to or remove information from the cell state, but it's designed to allow important information to be preserved over long distances. Think of it as the core context that the LSTM maintains.

-   **Hidden State (`h_t`):** This is the **output** of the LSTM unit at a particular time step and is often considered the **short-term memory**. It is a filtered version of the cell state, regulated by the output gate. The hidden state is what is passed to the next layer in a stacked LSTM or to a final classifier. It's the "working memory" that is used to make a prediction at the current time step.

In summary: The cell state carries the main information, while the hidden state is a filtered version of that information used for the output at that specific time step.

### 3. When building an RNN for text classification, you get two main outputs from an `nn.LSTM` layer: `output` and `(h_n, c_n)`. Explain what each of these represents and which one is typically used for classification.

**Answer:**
Let's assume `batch_first=True`.

-   **`output`**: This is a tensor of shape `(batch_size, sequence_length, hidden_size)`. It contains the **hidden state from the final LSTM layer for every single time step** in the sequence. `output[:, 0, :]` would be the hidden state after the first word, `output[:, 1, :]` after the second, and so on.

-   **`(h_n, c_n)`**: This is a tuple containing the final states of the LSTM.
    -   `h_n`: The **final hidden state** for every layer in the LSTM. Its shape is `(num_layers, batch_size, hidden_size)`. `h_n[-1]` gives you the hidden state of the *last* layer after the *last* time step.
    -   `c_n`: The **final cell state** for every layer. It has the same shape as `h_n`.

**Which one is used for classification?**
For a standard text classification task (like sentiment analysis), you typically want a single vector that summarizes the entire sentence. You can get this vector in two common ways:
1.  **Using `h_n` (Most Common):** The most common approach is to take the **final hidden state of the last layer**, which is `h_n[-1]`. This vector is considered a summary of the entire sequence and is passed to the final fully connected layer for classification.
2.  **Using `output`:** You can also take the hidden state of the last time step from the `output` tensor, which is `output[:, -1, :]`. For a single-layer LSTM, this is identical to `h_n[-1]`.

Therefore, you would typically use either `h_n[-1]` or `output[:, -1, :]` as the input to your final `nn.Linear` classifier.
