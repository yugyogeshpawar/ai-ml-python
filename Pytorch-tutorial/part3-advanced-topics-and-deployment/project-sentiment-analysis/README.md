# Part 3 Project: Sentiment Analysis with a Pre-trained Transformer

This project demonstrates a modern and powerful approach to NLP by using a pre-trained Transformer model for sentiment analysis. We will fine-tune a `DistilBERT` model on a dataset and then serve it as a simple local API using FastAPI.

This project leverages the Hugging Face `transformers` and `datasets` libraries, which are the industry standard for working with state-of-the-art NLP models in PyTorch.

## Goal

The goal is to build a complete, end-to-end NLP pipeline:
1.  Load a pre-trained Transformer model and a standard sentiment analysis dataset.
2.  Fine-tune the model on the dataset.
3.  Create a simple web API that can take a sentence as input and return a sentiment prediction.

## The `transformers` Library

Hugging Face's `transformers` library provides an easy-to-use interface for thousands of pre-trained models like BERT, GPT, and T5. It handles the complexity of tokenization and model loading, allowing us to focus on the application.

## Project Steps

This project is broken into two main scripts:

### 1. `sentiment_analysis.py`: Fine-Tuning the Model

This script handles the training part of the project.
-   **Load Dataset:** It loads the `imdb` dataset from the Hugging Face `datasets` library. This is a classic movie review dataset for binary sentiment classification.
-   **Load Tokenizer:** It loads the pre-trained `DistilBERT` tokenizer. The tokenizer is responsible for converting raw text into numerical IDs that the model can understand.
-   **Tokenize Data:** The script tokenizes the dataset, preparing the `input_ids` and `attention_mask` needed by the model.
-   **Load Model:** It loads a pre-trained `DistilBERT` model with a sequence classification head (`AutoModelForSequenceClassification`). The head is a new, untrained linear layer on top of the pre-trained model, ready for fine-tuning.
-   **Fine-Tuning:** It uses the `Trainer` API from the `transformers` library. The `Trainer` abstracts away the manual training loop, making it easy to fine-tune a model with best practices (like learning rate scheduling) built-in.
-   **Save Model:** After training, the fine-tuned model and its tokenizer are saved to a local directory (`./sentiment_model`).

### 2. `serve_with_fastapi.py`: Deploying the Model as an API

This script takes the fine-tuned model and serves it.
-   **Install FastAPI:** You will need to install `fastapi` and an ASGI server like `uvicorn`.
    ```bash
    pip install fastapi "uvicorn[standard]"
    ```
-   **Load Fine-Tuned Model:** The script loads the model and tokenizer that were saved by the training script.
-   **Create FastAPI App:** It creates a simple FastAPI web application.
-   **Define Prediction Endpoint:** It defines a `/predict/` endpoint that accepts a sentence as input.
-   **Inference Logic:** When a request is received, the script tokenizes the input sentence, passes it through the model, and converts the output logits into a "Positive" or "Negative" prediction.
-   **Run the Server:** The API can be run from the terminal using `uvicorn`.
    ```bash
    uvicorn serve_with_fastapi:app --reload
    ```
    You can then send requests to `http://127.0.0.1:8000/predict/` to get sentiment predictions.

This project represents a complete, modern workflow for building and deploying a powerful NLP model.
