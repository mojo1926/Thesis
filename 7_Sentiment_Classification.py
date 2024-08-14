import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define file paths
input_file_path = 'Risks/Risk_Comparison.xlsx'
labeled_risks_output_path = 'Risks/labeled_risks.xlsx'
model_path = 'Trained_Model/FINBERT_Service_3_epochs_9_layers_frozen_epoch_2.pth'
tokenizer_path = 'Trained_Model/FINBERT_Service_3_epochs_9_layers_frozen_tokenizer'

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Load the tokenizer from the directory containing all the necessary files
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=2, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)  # Move the model to the GPU
model.eval()

def predict_and_categorize_sentiment(text):
    if not isinstance(text, str):
        text = str(text)
        
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the GPU

    # Perform model inference and get logits
    with torch.no_grad():
        logits = model(**inputs).logits

    # Calculate probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()  # Move logits back to CPU

    # Calculate sentiment score (positive probability - negative probability)
    sentiment_score = probabilities[0][1] - probabilities[0][0]

    # Categorize sentiment based on the score
    categorized_sentiment = 1 if sentiment_score > 0 else -1

    return categorized_sentiment

def sentiment_analysis(df, column_name):
    sentiments = df[column_name].apply(predict_and_categorize_sentiment)
    return sentiments

def label_and_save_sentiments(input_file_path, output_file_path):
    # Load data from the Excel file
    df = pd.read_excel(input_file_path, sheet_name=None)
    
    # Perform sentiment analysis
    df['Identified Risks']['Sentiment'] = sentiment_analysis(df['Identified Risks'], 'Identified Risks')
    df['Random Risks']['Sentiment'] = sentiment_analysis(df['Random Risks'], 'Random Risks')
    df['Filed Risks']['Sentiment'] = sentiment_analysis(df['Filed Risks'], 'Risks')
    
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, data in df.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Sentiments labeled and saved to {output_file_path}")

# Run the sentiment labeling and save to Excel
label_and_save_sentiments(input_file_path, labeled_risks_output_path)
