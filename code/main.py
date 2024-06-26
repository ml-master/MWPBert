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

# 保存训练和验证图表的函数
# 该函数会创建并保存训练和验证准确率和损失的图表。
def save_plots(train_acc, train_loss, val_acc, val_loss, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘制并保存训练和验证准确率
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

    # 绘制并保存训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()




def extract_sentences(text):
    return sent_tokenize(text)


def get_longest_span(sentences, start, max_length):
    span = []
    current_length = 0
    for i in range(start, len(sentences)):
        sentence_length = len(sentences[i].split())
        if current_length + sentence_length <= max_length:
            span.append(sentences[i])
            current_length += sentence_length
        else:
            break
    return span


def average_score(span):
    if len(span) == 0:
        return 0  # 返回0以避免除以0错误
    scores = [random.random() for _ in span]
    return sum(scores) / len(scores)


def maxworth_algorithm(text, max_length=512):
    sentences = extract_sentences(text)
    start = 0
    max_score = 0
    max_span = None

    while start < len(sentences):
        span = get_longest_span(sentences, start, max_length)
        score = average_score(span)
        if score > max_score:
            max_score = score
            max_span = span
        start += 1

    return " ".join(max_span) if max_span else ""  # 确保返回的字符串不是空的


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


def create_data_loader(dataframe, tokenizer, max_len, batch_size):
    dataset = TextPairDataset(dataframe, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)


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


def load_data(dataset_file):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna().reset_index(drop=True)  # Remove missing values and reset index

    # 使用MaxWorth算法选择最有价值的片段
    df['text_1'] = df['text_1'].apply(lambda x: maxworth_algorithm(x, max_length=512))
    df['text_2'] = df['text_2'].apply(lambda x: maxworth_algorithm(x, max_length=512))

    return df


def init_config():
    json_object = json.load(open("config.json"))

    config = {
        "dataset_file": json_object["dataset_file"],
        "dump_location": json_object["dump_location"],
        "num_process": int(json_object["num_process"])
    }

    return config


def init_logging():
    format = '%(asctime)s %(process)d %(module)s %(levelname)s %(message)s'
    logging.basicConfig(
        filename='data_collection_{}.log'.format(str(int(time.time()))),
        level=logging.INFO,
        format=format)
    logging.getLogger('requests').setLevel(logging.CRITICAL)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

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

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


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



def main():
    # 初始化配置和日志记录
    config = init_config()
    init_logging()

    # 加载数据
    df = load_data(config["dataset_file"])

    # 拆分数据
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    # 加载BERT分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DualBERTModel('bert-base-uncased', num_labels=2)

    # 创建数据加载器
    train_data_loader = create_data_loader(df_train, tokenizer, max_len=128, batch_size=16)
    val_data_loader = create_data_loader(df_val, tokenizer, max_len=128, batch_size=16)

    # 设置训练参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-6, correct_bias=False)
    total_steps = len(train_data_loader) * 10
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # 初始化列表以存储损失和准确率值
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    # 训练模型
    for epoch in range(5):
        print(f'Epoch {epoch + 1}/{5}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )

        train_accuracies.append(train_acc.cpu().numpy())  # 将张量移动到CPU并转换为NumPy数组
        train_losses.append(train_loss)
        val_accuracies.append(val_acc.cpu().numpy())  # 将张量移动到CPU并转换为NumPy数组
        val_losses.append(val_loss)

        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

    # 保存图表
    save_plots(train_accuracies, train_losses, val_accuracies, val_losses, config["dump_location"])

    # 手动保存模型状态字典
    model_save_path = os.path.join(config["dump_location"], 'dual-bert-fake-news-model')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))

    # 保存分词器
    tokenizer.save_pretrained(os.path.join(config["dump_location"], 'dual-bert-fake-news-tokenizer'))

if __name__ == "__main__":
    main()
