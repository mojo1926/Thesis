import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import gc

# Define models and tokenizers
models = [
    BertForSequenceClassification.from_pretrained('nlpaueb/sec-bert-base', num_labels=2, ignore_mismatched_sizes=True),
    AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=2, ignore_mismatched_sizes=True),
    AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2, ignore_mismatched_sizes=True)
]

tokenizers = [
    BertTokenizer.from_pretrained('nlpaueb/sec-bert-base'),
    AutoTokenizer.from_pretrained('ProsusAI/finbert'),
    AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
]

# Define datasets
training_datasets = [
    '/content/drive/MyDrive/Thesis/10K_Sample_Files/Risks_for_Companies/extracted_risks.xlsx', 
    'Data/All_Service_Risks.xlsx'
]

model_names_path = ['SEC-BERT', 'FINBERT','BERT-Base']
sector_names_path = ['Real-Estate', 'Service']

# Choose model and dataset
model_number = 0  # choose 0 for SEC-BERT, 1 for FINBERT, 2 for BERT-Base
sector_training_data = 1  # 0 for Real Estate, 1 for Service sector data

training_dataset = training_datasets[sector_training_data]
model_name = models[model_number]
model_name_path = model_names_path[model_number]
sector_name_path = sector_names_path[sector_training_data]
tokenizer_name = tokenizers[model_number]

# Set parameters
num_epochs = 3
frozen_layers = 9
model_save_name = f"{model_name_path}_{sector_name_path}_{num_epochs}_epochs_{frozen_layers}_layers_frozen"

print(training_dataset)
print(model_name)
print(tokenizer_name)
print(model_save_name)

def train_and_save_model(excel_file_path, tokenizer_name, model_name, frozen_layers, num_epochs, model_save_name):
    print("Starting training...")
    # Load the dataset from an Excel file
    df = pd.read_excel(excel_file_path)

    # Encode the Sentiment labels
    label_encoder = LabelEncoder()
    df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])  # Convert 'Positive' and 'Negative' to numerical labels

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sophisticated split
    # Training split
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['Risks'], df['Sentiment'], test_size=0.1, shuffle=True, random_state=42)
    # Split the dataset into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, shuffle=True, random_state=42)

    # Initialize the tokenizer
    tokenizer = tokenizer_name

    # Tokenize the texts
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, return_tensors="pt")
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, return_tensors="pt")

    # Convert labels to tensors with correct data type
    train_labels = torch.tensor(train_labels.values, dtype=torch.long)
    val_labels = torch.tensor(val_labels.values, dtype=torch.long)
    test_labels = torch.tensor(test_labels.values, dtype=torch.long)

    # Create TensorDatasets for both training and validation sets
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

    # Load model
    model = model_name

    # Freeze layers
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    for i in range(frozen_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size_1 = 8

    # Prepare DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size_1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_1)

    # Initialize optimizer with a scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

    # Tracking variables
    epoch_losses = []
    val_losses = []
    learning_rates = []
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}
    test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}

    # Prepare DataFrame to save metrics
    metrics_df = pd.DataFrame(columns=['epoch', 'model_name', 'tokenizer_name', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_specificity'])

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}.")
        model.train()
        total_loss = 0  # Accumulate losses for each batch
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

        epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        model.eval()
        total_val_loss = 0
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Track learning rate
        learning_rates.append(scheduler.get_last_lr()[0])

        # Evaluate the model
        accuracy, precision, recall, f1, specificity = evaluate_model(model, val_loader, device)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['specificity'].append(specificity)

        test_accuracy, test_precision, test_recall, test_f1, test_specificity = evaluate_model(model, test_loader, device)
        test_metrics['accuracy'].append(test_accuracy)
        test_metrics['precision'].append(test_precision)
        test_metrics['recall'].append(test_recall)
        test_metrics['f1'].append(test_f1)
        test_metrics['specificity'].append(test_specificity)

        # Save metrics for this epoch to the DataFrame
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'model_name': [model_save_name],
            'tokenizer_name': [tokenizer_name.__class__.__name__],
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'specificity': [specificity],
            'test_accuracy': [test_accuracy],
            'test_precision': [test_precision],
            'test_recall': [test_recall],
            'test_f1': [test_f1],
            'test_specificity': [test_specificity]
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        # Save model weights for each epoch
        epoch_model_save_path = f'C:/Users/antho/OneDrive/Masterthesis/Thesis/Trained_Model/{model_save_name}_epoch_{epoch}.pth'
        torch.save(model.state_dict(), epoch_model_save_path)
        print(f"Model for epoch {epoch} saved to {epoch_model_save_path}")

    # Save the final model and tokenizer
    model_save_path = f'C:/Users/antho/OneDrive/Masterthesis/Thesis/Trained_Model/{model_save_name}_model.pth'
    tokenizer_save_path = f'C:/Users/antho/OneDrive/Masterthesis/Thesis/Trained_Model/{model_save_name}_tokenizer'
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    print(f"Final model saved to {model_save_path}")
    print(f"Tokenizer saved to {tokenizer_save_path}")

    # Save metrics and metadata to Excel file
    metrics_save_path = f'C:/Users/antho/OneDrive/Masterthesis/Thesis/Trained_Model/{model_save_name}_metrics.xlsx'
    
    # Metadata information
    metadata = {
        'Model Name': model_save_name,
        'Tokenizer Name': tokenizer_name.__class__.__name__,
        'Number of Epochs': num_epochs,
        'Frozen Layers': frozen_layers,
        'Training Dataset': excel_file_path,
    }
    
    # Convert metadata to DataFrame
    metadata_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])
    
    with pd.ExcelWriter(metrics_save_path) as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"Metrics and metadata saved to {metrics_save_path}")

    visualize_training(epoch_losses, val_losses, learning_rates, metrics, test_metrics, range(1, num_epochs + 1))

    # Clear variables to free up RAM
    del model, optimizer, scheduler, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_encodings, val_encodings, test_encodings
    gc.collect()
    torch.cuda.empty_cache()

def evaluate_model(model, loader, device):
    model.eval()  # Put the model in evaluation mode
    true_labels = []
    predictions = []

    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        true_labels.extend(batch[2].cpu().numpy())
        predictions.extend(preds.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    specificity = tn / (tn + fp)

    return accuracy, precision, recall, f1, specificity

def visualize_training(epoch_losses, val_losses, learning_rates, metrics, test_metrics, epochs):
    plt.figure(figsize=(18, 9))

    # Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, epoch_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Learning Rate
    plt.subplot(2, 2, 2)
    plt.plot(epochs, learning_rates, 'go-', label='Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    # Validation Metrics
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['accuracy'], 'r-o', label='Accuracy')
    plt.plot(epochs, metrics['precision'], 'g-o', label='Precision')
    plt.plot(epochs, metrics['recall'], 'b-o', label='Recall')
    plt.plot(epochs, metrics['f1'], 'y-o', label='F1 Score')
    plt.plot(epochs, metrics['specificity'], 'm-o', label='Specificity')
    plt.title('Validation Performance over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    # Test Metrics
    plt.subplot(2, 2, 4)
    plt.plot(epochs, test_metrics['accuracy'], 'r-o', label='Accuracy')
    plt.plot(epochs, test_metrics['precision'], 'g-o', label='Precision')
    plt.plot(epochs, test_metrics['recall'], 'b-o', label='Recall')
    plt.plot(epochs, test_metrics['f1'], 'y-o', label='F1 Score')
    plt.plot(epochs, test_metrics['specificity'], 'm-o', label='Specificity')
    plt.title('Test Performance over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Train and save the model
trained_model = train_and_save_model(training_dataset, tokenizer_name, model_name, frozen_layers, num_epochs, model_save_name)
