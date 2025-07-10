# LLM Tutorials

## 01 Introduction to Large Language Models (LLMs)

### Introduction

Welcome to the first tutorial on Large Language Models (LLMs)! This tutorial will provide you with a foundational understanding of LLMs, their importance, and key concepts.

## What are LLMs?

Large Language Models (LLMs) are a type of artificial intelligence (AI) model that can understand and generate human-like text. They are trained on massive datasets of text and code, allowing them to perform a wide range of natural language processing (NLP) tasks.

## Why are LLMs important?

LLMs are revolutionizing various fields due to their ability to:

*   **Generate human-quality text:** Create articles, stories, poems, and more.
*   **Understand and respond to complex queries:** Answer questions, provide summaries, and engage in conversations.
*   **Automate tasks:** Translate languages, write code, and perform other tasks that previously required human intervention.

## Key Concepts

*   **Tokens:** The basic units of text that LLMs process. Text is broken down into tokens (words, parts of words, or characters).
*   **Embeddings:** Numerical representations of words or tokens that capture their meaning and relationships.
*   **Attention Mechanism:** A core component of LLMs that allows the model to focus on the most relevant parts of the input when generating output.

## Different Types of LLMs

*   **Generative Pre-trained Transformers (GPT):** Known for their text generation capabilities.
*   **Bidirectional Encoder Representations from Transformers (BERT):** Designed for understanding context and relationships in text.
*   **Other LLMs:** There are many other LLMs, each with its strengths and weaknesses.

## Use Cases of LLMs

*   **Chatbots and Conversational AI:** Creating interactive and engaging conversational experiences.
*   **Content Creation:** Generating articles, blog posts, and social media content.
*   **Language Translation:** Translating text between different languages.
*   **Code Generation:** Assisting developers in writing code.
*   **Text Summarization:** Condensing long texts into shorter summaries.

## Assignment

Research and summarize a recent application of LLMs. Provide details on the LLM used, the task performed, and the results achieved.

## Interview Question

Explain the basic architecture of an LLM.

## Exercises

1.  **Define LLMs:** In your own words, explain what Large Language Models (LLMs) are and what they are used for.
2.  **Identify Key Concepts:** Briefly describe the following key concepts related to LLMs: tokens, embeddings, and the attention mechanism.
3.  **Research Use Cases:** Find three different real-world applications of LLMs and briefly describe how they are being used.
4.  **Architecture Overview:** Draw a simplified diagram of an LLM architecture, labeling the key components.

## 02 Open Source LLMs

### Introduction

This tutorial explores Open Source Large Language Models (LLMs), their advantages, disadvantages, and how to use them.

## Overview of Open Source LLMs

Open Source LLMs are language models whose code and weights are publicly available, allowing for community contributions, modifications, and redistribution. Examples include:

*   Llama (Meta)
*   Mistral (Mistral AI)
*   Falcon (Technology Innovation Institute)
*   Bloom (BigScience)

## Advantages of Open Source LLMs

*   **Transparency:** Code and model weights are open for inspection and auditing.
*   **Customization:** Models can be fine-tuned and adapted to specific tasks and datasets.
*   **Community Support:** Active communities provide resources, support, and updates.
*   **Cost-effective:** Often free to use and distribute.

## Disadvantages of Open Source LLMs

*   **Resource Intensive:** Training and fine-tuning can require significant computational resources.
*   **Model Quality:** Performance may vary compared to closed-source models, depending on the specific model and task.
*   **Licensing:** Understanding and adhering to the specific licenses (e.g., Apache 2.0, MIT) is crucial.

## How to Access and Use Open Source LLMs

*   **Hugging Face Hub:** A popular platform for accessing and sharing pre-trained models and datasets.
*   **Model APIs:** Some providers offer APIs for easy access to their models.
*   **Local Deployment:** Models can be downloaded and run locally on your hardware.

## Code Example: Using a pre-trained Open Source LLM for text generation (Python with Transformers)

```python
from transformers import pipeline

# Choose a model (e.g., a model from Hugging Face Hub)
model_name = "gpt2"  # Replace with the actual model name

# Create a text generation pipeline
generator = pipeline("text-generation", model=model_name)

# Generate text
prompt = "The meaning of life is"
output = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(output[0]["generated_text"])
```

## Assignment

Experiment with different Open Source LLMs and compare their performance on a specific task (e.g., text summarization, question answering, code generation). Document your findings, including the models used, the task, the evaluation metrics, and your conclusions.

## Interview Question

What are the benefits of using Open Source LLMs?

## Exercises

1.  **Research Open Source LLMs:** List three different Open Source LLMs and briefly describe their key features.
2.  **Licensing:** Explain the importance of understanding the license of an Open Source LLM. Provide examples of different licenses and their implications.
3.  **Code Exploration:** Choose one of the Open Source LLMs and find a code example (e.g., on Hugging Face Hub) that demonstrates how to use it for a specific task. Explain the code.
4.  **Compare and Contrast:** Compare and contrast the advantages and disadvantages of Open Source LLMs.

## 03 Closed Source LLMs

### Introduction

This tutorial covers Closed Source Large Language Models (LLMs), their characteristics, and how they compare to Open Source LLMs.

## Overview of Closed Source LLMs

Closed Source LLMs are language models whose code and weights are not publicly available. They are typically developed and maintained by private companies. Examples include:

*   GPT-4 (OpenAI)
*   Gemini (Google)
*   Claude (Anthropic)

## Advantages of Closed Source LLMs

*   **High Performance:** Often achieve state-of-the-art results due to extensive resources and proprietary training data.
*   **Ease of Use:** Typically offer user-friendly APIs and interfaces.
*   **Reliability and Support:** Backed by dedicated teams and infrastructure.

## Disadvantages of Closed Source LLMs

*   **Lack of Transparency:** Limited access to model details and training data.
*   **Cost:** Usage often involves subscription fees or pay-per-use pricing.
*   **Limited Customization:** Fine-tuning options may be restricted.
*   **Vendor Lock-in:** Dependence on a specific provider.

## Accessing and Using Closed Source LLMs

*   **APIs:** Most closed-source LLMs are accessed through APIs (e.g., OpenAI API, Google AI Platform).
*   **SDKs:** Software Development Kits (SDKs) may be available to simplify integration.
*   **Web Interfaces:** Some providers offer web-based interfaces for interacting with their models.

## Code Example: Using a Closed Source LLM API for text summarization (Python with API calls - Example using OpenAI)

```python
import openai

# Set your API key
openai.api_key = "YOUR_API_KEY"  # Replace with your actual API key

def summarize_text(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Or another suitable engine
            prompt=f"Summarize the following text:\n{text}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
text_to_summarize = """
Large Language Models (LLMs) are a type of artificial intelligence (AI) model that can understand and generate human-like text. They are trained on massive datasets of text and code, allowing them to perform a wide range of natural language processing (NLP) tasks. LLMs are revolutionizing various fields due to their ability to generate human-quality text, understand and respond to complex queries, and automate tasks.
"""
summary = summarize_text(text_to_summarize)
print(summary)
```

## Assignment

Compare the performance of an Open Source LLM and a Closed Source LLM on a specific task, considering factors like cost, speed, and accuracy. Document your findings, including the models used, the task, the evaluation metrics, and your conclusions.

## Interview Question

What are the trade-offs between Open Source and Closed Source LLMs?

## Exercises

1.  **Research Closed Source LLMs:** List three different Closed Source LLMs and briefly describe their key features and pricing models.
2.  **API Exploration:** Choose one Closed Source LLM and explore its API documentation. Describe the available features and limitations.
3.  **Cost Analysis:** Compare the cost of using a Closed Source LLM API with the cost of running an Open Source LLM on your own hardware (consider factors like hardware costs, electricity, and maintenance).
4.  **Ethical Considerations:** Discuss the ethical implications of using Closed Source LLMs, considering issues like data privacy, bias, and transparency.

## 04 Transformers: The Engine Behind LLMs

### Introduction

This tutorial delves into the Transformer architecture, the core building block of modern LLMs.

## A Deeper Dive into the Transformer Architecture

The Transformer architecture is the foundation of modern LLMs. It relies on the attention mechanism to process input sequences and generate output. Unlike recurrent neural networks (RNNs), Transformers process the entire input sequence in parallel, enabling faster training and better performance.

## Attention Mechanism Explained in Detail

The attention mechanism allows the model to weigh the importance of different words in the input sequence when generating the output. It works by:

1.  **Calculating Attention Scores:** For each word in the input, calculate a score representing its relevance to other words.
2.  **Applying Softmax:** Normalize the attention scores using the softmax function to create a probability distribution.
3.  **Weighted Sum:** Calculate a weighted sum of the input embeddings, where the weights are the attention probabilities.

## Encoder and Decoder Components

The Transformer architecture typically consists of two main components:

*   **Encoder:** Processes the input sequence and creates a contextualized representation.
*   **Decoder:** Generates the output sequence based on the encoder's output and the attention mechanism.

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

## 05 Fine-tuning LLMs

### Introduction

This tutorial covers the process of fine-tuning Large Language Models (LLMs) to adapt them to specific tasks.

## What is Fine-tuning?

Fine-tuning is the process of taking a pre-trained LLM and further training it on a specific dataset to adapt it to a particular task or domain. This allows you to leverage the knowledge learned by the pre-trained model while tailoring it to your specific needs.

## Why Fine-tune LLMs?

*   **Improved Performance:** Fine-tuning can significantly improve the performance of an LLM on a specific task compared to using the pre-trained model directly.
*   **Domain Adaptation:** Fine-tuning allows you to adapt an LLM to a specific domain or style of writing.
*   **Task Specialization:** Fine-tuning enables you to specialize an LLM for tasks like question answering, text summarization, or code generation.

## Different Fine-tuning Techniques

*   **Full Fine-tuning:** Training all the parameters of the LLM. This can be computationally expensive.
*   **Parameter-Efficient Fine-tuning (PEFT):** Techniques that fine-tune only a small subset of the model's parameters, reducing computational cost. Examples include:
    *   **LoRA (Low-Rank Adaptation):** Adds trainable low-rank matrices to the model's layers.
    *   **Prefix Tuning:** Adds a trainable prefix to the input sequence.
    *   **Prompt Tuning:** Optimizes a set of continuous prompts.

## Data Preparation for Fine-tuning

*   **Dataset Selection:** Choose a dataset relevant to your target task or domain.
*   **Data Cleaning:** Clean and preprocess the data to ensure quality.
*   **Data Formatting:** Format the data in a way that is compatible with the LLM and the fine-tuning process. This often involves creating input-output pairs.

## Code Example: Fine-tuning a pre-trained LLM on a custom dataset (Python with Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load the pre-trained model and tokenizer
model_name = "gpt2"  # Replace with the desired model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Prepare the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128) # Adjust max_length as needed

# Create a dummy dataset (replace with your actual dataset)
data = [{"text": "This is a sample sentence."}, {"text": "Another example sentence."}]
dataset = Dataset.from_dict({"text": [item["text"] for item in data]})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
)

# 4. Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 5. Fine-tune the model
trainer.train()

# 6. Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
```

## Assignment

Fine-tune an LLM on a dataset of your choice (e.g., a dataset for question answering, text summarization, or code generation) and evaluate its performance. Compare the results to the pre-trained model.

## Interview Question

What are the key considerations when fine-tuning an LLM?

## Exercises

1.  **Define Fine-tuning:** Explain what fine-tuning is and why it is used.
2.  **PEFT Techniques:** Research and compare different Parameter-Efficient Fine-tuning (PEFT) techniques (e.g., LoRA, Prefix Tuning, Prompt Tuning).
3.  **Dataset Selection:** Describe the factors to consider when selecting a dataset for fine-tuning an LLM.
4.  **Code Modification:** Modify the provided code example to fine-tune the LLM on a different dataset or for a different task.
