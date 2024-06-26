import json
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import json
import logging
import time
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from torch import nn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import random

import torch

import matplotlib.pyplot as plt

class TextPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            text1 = str(self.data.iloc[index].text_1)
            text2 = str(self.data.iloc[index].text_2)
            label = self.data.iloc[index].label

            encoding1 = self.tokenizer.encode_plus(
                text1,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            encoding2 = self.tokenizer.encode_plus(
                text2,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            return {
                'input_ids_1': encoding1['input_ids'].flatten(),
                'attention_mask_1': encoding1['attention_mask'].flatten(),
                'token_type_ids_1': encoding1['token_type_ids'].flatten(),
                'input_ids_2': encoding2['input_ids'].flatten(),
                'attention_mask_2': encoding2['attention_mask'].flatten(),
                'token_type_ids_2': encoding2['token_type_ids'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise

class DualBERTModel(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(DualBERTModel, self).__init__()
        self.bert1 = BertModel.from_pretrained(pretrained_model_name)
        self.bert2 = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert1.config.hidden_size * 2, num_labels)

    def forward(self, input_ids_1, attention_mask_1, token_type_ids_1,
                input_ids_2, attention_mask_2, token_type_ids_2):
        output1 = self.bert1(input_ids=input_ids_1,
                             attention_mask=attention_mask_1,
                             token_type_ids=token_type_ids_1)

        output2 = self.bert2(input_ids=input_ids_2,
                             attention_mask=attention_mask_2,
                             token_type_ids=token_type_ids_2)

        pooled_output1 = output1.pooler_output
        pooled_output2 = output2.pooler_output

        combined_output = torch.cat((pooled_output1, pooled_output2), dim=1)
        combined_output = self.dropout(combined_output)
        logits = self.classifier(combined_output)

        return logits

def create_data_loader(dataframe, tokenizer, max_len, batch_size):
    dataset = TextPairDataset(dataframe, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids_1 = d["input_ids_1"].to(device)
            attention_mask_1 = d["attention_mask_1"].to(device)
            token_type_ids_1 = d["token_type_ids_1"].to(device)
            input_ids_2 = d["input_ids_2"].to(device)
            attention_mask_2 = d["attention_mask_2"].to(device)
            token_type_ids_2 = d["token_type_ids_2"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids_1=input_ids_1,
                attention_mask_1=attention_mask_1,
                token_type_ids_1=token_type_ids_1,
                input_ids_2=input_ids_2,
                attention_mask_2=attention_mask_2,
                token_type_ids_2=token_type_ids_2
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def load_data(dataset_file):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna().reset_index(drop=True)  # Remove missing values and reset index
    return df

def save_test_results(output_dir, results):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'test_results.txt'), 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')

def main():
    # Load config
    config = {
        "dataset_file": "../dataset/train.json",  # Update with actual path
        "model_dir": "./model_output",  # Update with actual path
        "output_dir": "./model_output",  # Update with actual path
        "max_len": 128,
        "batch_size": 16
    }

    # Load data
    df = load_data(config["dataset_file"])
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(config["model_dir"] + '/dual-bert-fake-news-tokenizer')
    model = DualBERTModel('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(config["model_dir"] + '/dual-bert-fake-news-model/pytorch_model.bin'))

    # Create data loader
    test_data_loader = create_data_loader(df_test, tokenizer, config["max_len"], config["batch_size"])

    # Evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device, len(df_test))
    print(f'Test loss {test_loss} accuracy {test_acc}')

    # Save evaluation results
    results = [f'Test loss: {test_loss}', f'Test accuracy: {test_acc}']

    # Test a few examples
    model.eval()
    with torch.no_grad():
        for i in range(5):  # Test 5 examples
            sample = df_test.sample(1).iloc[0]
            text1, text2, label = sample['text_1'], sample['text_2'], sample['label']

            encoding1 = tokenizer.encode_plus(
                text1,
                add_special_tokens=True,
                max_length=config["max_len"],
                padding='max_length',
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            encoding2 = tokenizer.encode_plus(
                text2,
                add_special_tokens=True,
                max_length=config["max_len"],
                padding='max_length',
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            input_ids_1 = encoding1['input_ids'].to(device)
            attention_mask_1 = encoding1['attention_mask'].to(device)
            token_type_ids_1 = encoding1['token_type_ids'].to(device)
            input_ids_2 = encoding2['input_ids'].to(device)
            attention_mask_2 = encoding2['attention_mask'].to(device)
            token_type_ids_2 = encoding2['token_type_ids'].to(device)

            outputs = model(
                input_ids_1=input_ids_1,
                attention_mask_1=attention_mask_1,
                token_type_ids_1=token_type_ids_1,
                input_ids_2=input_ids_2,
                attention_mask_2=attention_mask_2,
                token_type_ids_2=token_type_ids_2
            )

            _, pred = torch.max(outputs, dim=1)

            result = f'Text 1: {text1}\nText 2: {text2}\nTrue Label: {label}, Predicted Label: {pred.item()}\n---'
            print(result)
            results.append(result)

    # Save test results to file
    save_test_results(config["output_dir"], results)

if __name__ == "__main__":
    main()
