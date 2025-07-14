# serve_with_fastapi.py

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Setup ---
# This script should be run after `sentiment_analysis.py` has been executed
# and the model has been saved.

# Define the application
app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API to predict the sentiment of a given text.",
    version="1.0",
)

# --- 2. Load the Fine-Tuned Model and Tokenizer ---
# Load the model and tokenizer from the directory where they were saved.
try:
    MODEL_DIR = "./sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"--- Model and tokenizer loaded successfully from {MODEL_DIR} ---")
    print(f"--- API is running on device: {device} ---")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run `sentiment_analysis.py` first to train and save the model.")
    model = None
    tokenizer = None

# --- 3. Define the Request and Response Models ---
# Pydantic models for type validation and API documentation
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float

# --- 4. Define the Prediction Endpoint ---
@app.post("/predict/", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """
    Predicts the sentiment of a given text.
    
    - **text**: The input text to analyze.
    \f
    :param request: The request body with the text.
    :return: A JSON response with the original text, predicted sentiment, and confidence score.
    """
    if not model or not tokenizer:
        return {"error": "Model not loaded. Please check server logs."}

    # Tokenize the input text
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # The model outputs logits for two classes (negative, positive).
    # We apply softmax to get probabilities.
    scores = torch.softmax(logits, dim=1)[0]
    
    # Get the predicted class index
    predicted_class_id = torch.argmax(scores).item()
    
    # Get the label name ('NEGATIVE' or 'POSITIVE')
    sentiment = model.config.id2label[predicted_class_id]
    
    # Get the confidence score for the predicted class
    score = scores[predicted_class_id].item()

    return {
        "text": request.text,
        "sentiment": sentiment,
        "score": score
    }

# --- 5. Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Go to /docs for documentation."}

# --- How to Run This API ---
# 1. Make sure you have run `sentiment_analysis.py` first.
# 2. Install FastAPI and Uvicorn: `pip install fastapi "uvicorn[standard]"`
# 3. Run from your terminal: `uvicorn serve_with_fastapi:app --reload`
# 4. Open your browser to http://127.0.0.1:8000/docs to see the interactive API documentation.
