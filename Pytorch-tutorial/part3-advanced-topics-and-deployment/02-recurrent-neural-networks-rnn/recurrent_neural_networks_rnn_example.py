# recurrent_neural_networks_rnn_example.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# --- 1. Data Preparation ---
# For this example, we'll create a simple, synthetic dataset for sentiment analysis.
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Simple vocabulary builder
def build_vocab(texts):
    word_counts = Counter()
    for text in texts:
        for word in text.split():
            word_counts[word] += 1
    # Create a vocabulary mapping each word to an index
    # Reserve index 0 for padding and 1 for unknown words
    vocab = {word: i + 2 for i, (word, _) in enumerate(word_counts.items())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

# Function to convert text to a sequence of indices
def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

# --- 2. Define the RNN Model ---
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super(SimpleRNN, self).__init__()
        
        # Embedding layer converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer processes the sequence of embeddings
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            batch_first=True) # Important!
        
        # Fully connected layer for final classification
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: (batch_size, seq_length)
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_length, embedding_dim)
        
        # Pass embeddings to LSTM
        # We don't need the hidden state output, just the final output
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        # hidden shape: (n_layers, batch_size, hidden_dim)
        
        # We'll use the hidden state of the last time step for classification
        final_hidden_state = hidden[-1] # Get the last layer's hidden state
        # final_hidden_state shape: (batch_size, hidden_dim)
        
        return self.fc(final_hidden_state)

def main():
    # --- Setup ---
    # Sample Data
    texts = ["i love this movie", "this was an amazing film", "what a great experience",
             "i hated this film", "a terrible waste of time", "i would not recommend this"]
    labels = [1, 1, 1, 0, 0, 0] # 1 for positive, 0 for negative

    # Build vocabulary
    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    
    # Convert texts to sequences and pad them to the same length
    sequences = [text_to_sequence(text, vocab) for text in texts]
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.array([seq + [0] * (max_len - len(seq)) for seq in sequences])

    # Create dataset and dataloader
    dataset = TextDataset(torch.LongTensor(padded_sequences), torch.FloatTensor(labels))
    dataloader = DataLoader(dataset, batch_size=2)

    # --- Model, Loss, and Optimizer ---
    embedding_dim = 50
    hidden_dim = 100
    output_dim = 1 # Single output for binary classification
    n_layers = 1

    model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers)
    criterion = nn.BCEWithLogitsLoss() # Good for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    print("--- Starting Training ---")
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        for texts_batch, labels_batch in dataloader:
            optimizer.zero_grad()
            
            # Get model predictions
            predictions = model(texts_batch).squeeze(1)
            
            # Calculate loss
            loss = criterion(predictions, labels_batch)
            
            # Backpropagate and update weights
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print("--- Training Finished ---")

    # --- Inference ---
    print("\n--- Making Predictions ---")
    model.eval()
    with torch.no_grad():
        test_texts = ["this was a great film", "i did not like this movie"]
        for text in test_texts:
            seq = torch.LongTensor([text_to_sequence(text, vocab)])
            prediction = model(seq).squeeze(1)
            # Use sigmoid to convert logit to probability
            prob = torch.sigmoid(prediction)
            sentiment = "Positive" if prob.item() > 0.5 else "Negative"
            print(f"Text: '{text}' -> Prediction: {sentiment} (Prob: {prob.item():.2f})")

if __name__ == '__main__':
    main()
