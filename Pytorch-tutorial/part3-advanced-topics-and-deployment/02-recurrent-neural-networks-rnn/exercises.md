# Exercises: Recurrent Neural Networks

These exercises will help you understand the components and data flow of an RNN model.

## Exercise 1: The `nn.Embedding` Layer

**Task:** The `nn.Embedding` layer is a lookup table that stores embeddings for a fixed dictionary and size. Let's see how it works.

1.  Define a vocabulary size of 10 and an embedding dimension of 3.
2.  Create an `nn.Embedding` layer with this vocabulary size and dimension.
3.  Create a sample input tensor of word indices, e.g., `torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])`. This represents a batch of 2 sentences, each with 4 words.
4.  Pass this input tensor through the embedding layer.
5.  Print the shape of the output tensor. What does each dimension represent?

**Goal:** Understand that the embedding layer takes a tensor of indices and outputs a dense tensor where each index has been replaced by its corresponding embedding vector.

## Exercise 2: The `batch_first` Argument

**Task:** The `batch_first` argument in `nn.LSTM` is very important. Let's see what happens when you change it.

1.  Create an `nn.LSTM` layer with `input_size=10`, `hidden_size=20`, and `batch_first=True`.
2.  Create a random input tensor with the shape `(5, 3, 10)`, which corresponds to `(batch, seq_len, features)`.
3.  Pass the input through the LSTM and print the shape of the `output` and `hidden_state`.
4.  Now, create a second `nn.LSTM` layer, but this time with `batch_first=False` (the default).
5.  Create a new random input tensor with the shape `(3, 5, 10)`, which corresponds to `(seq_len, batch, features)`.
6.  Pass this new input through the second LSTM and print the shapes of its outputs.

**Goal:** See how the `batch_first` argument changes the expected input shape and the resulting output shape, which is a common source of errors when working with RNNs.

## Exercise 3: Stacking LSTM Layers

**Task:** You can create deeper RNNs by stacking layers.

1.  Modify the `SimpleRNN` model from the example to have **two** LSTM layers instead of one. You can do this by setting the `n_layers=2` argument in the `nn.LSTM` constructor.
2.  Instantiate the model and print it.
3.  In the `forward` pass of the model, what is the shape of the `hidden` state tensor that is output by the LSTM? (Hint: its first dimension will be `n_layers`).
4.  How would you get the hidden state of just the *last* layer to pass to the fully connected layer?

**Goal:** Understand how to create stacked RNNs and how to correctly select the final hidden state for classification.
