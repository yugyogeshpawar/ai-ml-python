# sentiment_analysis.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def main():
    """
    Fine-tunes a pre-trained DistilBERT model for sentiment analysis on the IMDB dataset.
    """
    # --- 1. Load Dataset and Tokenizer ---
    print("--- Loading IMDB dataset and DistilBERT tokenizer ---")
    
    # Load the IMDB dataset
    # The `datasets` library will download and cache it.
    imdb = load_dataset('imdb')

    # Load the tokenizer for 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # --- 2. Preprocess and Tokenize the Data ---
    def preprocess_function(examples):
        # Tokenize the texts, truncating them to the model's max input length.
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    print("\n--- Tokenizing dataset ---")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    # For this demo, we'll use a smaller subset of the data to speed up training.
    # In a real project, you would use the full dataset.
    train_dataset = tokenized_imdb['train'].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_imdb['test'].shuffle(seed=42).select(range(1000))

    # --- 3. Load the Pre-trained Model ---
    print("\n--- Loading pre-trained DistilBERT model ---")
    
    # Load DistilBERT with a sequence classification head.
    # `num_labels=2` for binary sentiment classification (positive/negative).
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # --- 4. Define Training Arguments and Trainer ---
    print("\n--- Setting up Trainer ---")
    
    # Define the directory to save the model checkpoints
    model_dir = "./sentiment_model_checkpoints"

    # TrainingArguments specifies all the hyperparameters for training.
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=2, # Number of training epochs
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch" # Evaluate at the end of each epoch
    )

    # The Trainer class handles the entire training and evaluation loop.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # --- 5. Fine-Tune the Model ---
    print("\n--- Starting Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Finished ---")

    # --- 6. Save the Fine-Tuned Model ---
    print("\n--- Saving Model ---")
    save_directory = "./sentiment_model"
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

if __name__ == '__main__':
    # This requires the following packages:
    # pip install torch transformers datasets scikit-learn
    main()
