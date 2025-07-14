# 04 Transformers: The Engine Behind LLMs

## Introduction

This tutorial delves into the Transformer architecture, the core building block of modern LLMs.

## A Deeper Dive into the Transformer Architecture

The Transformer architecture is the foundation of modern LLMs. It relies on the attention mechanism to process input sequences and generate output. Unlike recurrent neural networks (RNNs), Transformers process the entire input sequence in parallel, enabling faster training and better performance.

## Attention Mechanism Explained in Detail

The attention mechanism allows the model to weigh the importance of different words in the input sequence when generating the output. It works by:

1.  **Calculating Attention Scores:** For each word in the input, calculate a score representing its relevance to other words.
2.  **Applying Softmax:** Normalize the attention scores using the softmax function to create a probability distribution.
3.  **Weighted Sum:** Calculate a weighted sum of the input embeddings, where the weights are the attention probabilities.

## Encoder and Decoder Components

The Transformer architecture consists of two main components: the **Encoder** and the **Decoder**.

*   **The Encoder Stack:** The encoder's job is to "understand" the input sequence. It is a stack of identical layers, each containing two sub-layers:
    1.  A multi-head self-attention mechanism.
    2.  A simple, position-wise fully connected feed-forward network.
*   **The Decoder Stack:** The decoder's job is to generate the output sequence. It is also a stack of identical layers, but each layer has three sub-layers:
    1.  A masked multi-head self-attention mechanism (to prevent positions from attending to subsequent positions).
    2.  A multi-head attention mechanism that takes the output of the encoder stack as input.
    3.  A simple, position-wise fully connected feed-forward network.

### Visualizing the Transformer Architecture

Here is a simplified diagram of the Transformer architecture:

```
+-----------------+      +-----------------+
|   Input Text    |      |  Output Text    |
+-----------------+      +-----------------+
        |                      ^
        v                      |
+-----------------+      +-----------------+
| Input Embedding |      | Output Embedding|
+-----------------+      +-----------------+
        |                      ^
        v                      |
+-----------------+      +-----------------+
| Positional Enc. |      | Positional Enc. |
+-----------------+      +-----------------+
        |                      ^
        v                      |
+-----------------+      +-----------------+
|    Encoder      |----->|    Decoder      |
| (N layers)      |      | (N layers)      |
+-----------------+      +-----------------+
```

## Positional Encoding

Since Transformers do not inherently understand the order of words in a sequence, positional encoding is used to add information about the position of each word. This is typically done by adding a vector to each word embedding that encodes its position in the sequence.

## Code Example: Implementing a simplified Transformer layer (Python with PyTorch)

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear projections
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)

        # Concatenate heads and project
        output = output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.out_linear(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x

# Example usage:
embed_dim = 512
num_heads = 8
ff_dim = 2048
batch_size = 32
seq_len = 10

# Create a random input tensor
input_tensor = torch.randn(batch_size, seq_len, embed_dim)

# Create a Transformer block
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

# Pass the input through the Transformer block
output = transformer_block(input_tensor)

# Print the output shape
print(output.shape)  # Expected output: torch.Size([32, 10, 512])
```

## Assignment

Implement a simple attention mechanism from scratch. This could involve creating the attention scores, applying softmax, and calculating the weighted sum.

## Interview Question

Explain the role of the attention mechanism in Transformers.

## Exercises

1.  **Explain the Architecture:** Describe the main components of the Transformer architecture (encoder, decoder, attention mechanism, positional encoding).
2.  **Attention Mechanism:** Explain how the attention mechanism works, including the calculation of attention scores, the application of softmax, and the weighted sum.
3.  **Code Analysis:** Analyze the provided code example (or another Transformer implementation) and explain how the multi-head attention mechanism is implemented.
4.  **Positional Encoding:** Explain the purpose of positional encoding in Transformers and why it is necessary.
