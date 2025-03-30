# Class: Northwestern CS 461 Winter 2025
# ---------------------------------------

# Professor: David Demeter
# ---------------------------------------

# Contributers:
# ---------------------------------------
#   Raymond Gu
#   Maanvi Sarwadi

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import classification_report

def main():

    # Set up the random seed
    torch.manual_seed(0)

    # Check if we can use the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define all the hyper-parameters
    loadname = None
    batch_size = 16
    num_epochs = 20
    learning_rate = 3e-5

    # Intialize the model and create the batches
    model = BERT_MODEL().to(device)
    train_batches, valid_batches, test_batches = get_batches(batch_size)

    # Load the weights if provided, otherwise train the model
    if loadname is not None:
        print("Loading pretrained weights...\n")
        model.load_state_dict(torch.load(loadname))
    else:
        train(model, train_batches, valid_batches, num_epochs, learning_rate, device)
        model.load_state_dict(torch.load('model_weights.pth'))

    # Get a report on the performance (precison, recall, f1-score) of the model
    report = get_performance_report(model, test_batches, device)
    print(report)

class BERT_MODEL(nn.Module):
    def __init__(self):

        super(BERT_MODEL, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        logits = self.linear(cls_embeddings)

        return logits

def train(model, train_batches, valid_batches, num_epochs, learning_rate, device):

    # Set the model to training mode
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a list to store loss on validation set after each epoch
    valid_epoch_loss = []

    for epoch in range(num_epochs):

        for batch in train_batches:

            # Move all the tensors in the batch to the device
            tokens, attention_mask, labels = [tensor.to(device) for tensor in batch]

            # Get the logits, calculate the loss, and update the model parameters
            logits = model(tokens, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Zero out the gradients
            optimizer.zero_grad()

        # Compute the loss for the training and validation sets
        train_loss = test(model, train_batches, criterion, device)
        valid_loss = test(model, valid_batches, criterion, device)
        valid_epoch_loss.append(valid_loss)

        # If performance on the validation set doesn't improve, stop early
        if (epoch >= 1 and valid_epoch_loss[-1] >= valid_epoch_loss[-2]):
            break

        # If the model performance improves, save the weights
        torch.save(model.state_dict(), 'model_weights.pth')

        model.train()

def test(model, batches, criterion, device):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        for batch in batches:

            # Move all the tensors in the batch to the device
            tokens, attention_mask, labels = [tensor.to(device) for tensor in batch]

            # Get the logits and calculate the loss
            logits = model(tokens, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(batches)

    return avg_loss

def get_performance_report(model, batches, device):
    
    model.eval()

    model_predictions = []
    correct_labels = []

    with torch.no_grad():

        for batch in batches:

            # Move all the tensors in the batch to the device
            tokens, attention_mask, labels = [tensor.to(device) for tensor in batch]

            # Get the logits and make a predictions with them
            logits = model(tokens, attention_mask)
            preds = torch.argmax(logits, dim=1)

            # Keep track of the predictions and correct labels
            model_predictions.extend(preds.cpu().numpy())
            correct_labels.extend(labels.cpu().numpy())

    # Compute the F1 score; using 'weighted' average to account for class imbalance if needed
    report = classification_report(correct_labels, model_predictions, target_names=["real", "fake"])

    return report

def get_batches(batch_size):

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create the training, validation, and test datasets
    train_dataset = create_dataset(r'./Cleaned Data/train.xlsx', tokenizer)
    valid_dataset = create_dataset(r'./Cleaned Data/valid.xlsx', tokenizer)
    test_dataset = create_dataset(r'./Cleaned Data/test.xlsx', tokenizer)

    # Create the traing, validation, and test batches
    train_batches = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_batches = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_batches = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_batches, valid_batches, test_batches

def create_dataset(file_path, tokenizer):

    # Get the dataframe from the Excel file
    df = pd.read_excel(file_path)

    # Convert the labels to binary (0=Fake, 1=Real)
    label_map = {"real":0, "fake":1}
    df["Label"] = df["Label"].map(label_map)

    # Extract the descriptions and labels
    df_texts = [str(text) for text in df["Description"].tolist()]
    df_labels = df["Label"].tolist()

    # Tokenize all the descriptions
    output = tokenizer(df_texts, padding="max_length", max_length=200, truncation=True, return_tensors="pt")

    # Create the dataset
    tokens_tensor = output["input_ids"]
    attention_mask_tensor = output["attention_mask"]
    labels_tensor = torch.tensor(df_labels, dtype=torch.long)
    dataset = TensorDataset(tokens_tensor, attention_mask_tensor, labels_tensor)

    return dataset

if __name__ == "__main__":
    main()