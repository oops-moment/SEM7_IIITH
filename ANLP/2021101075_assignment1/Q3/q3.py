# -*- coding: utf-8 -*-
"""Transformer Language Model for ANLP Assignment"""

import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.nn.utils.rnn import pad_sequence

nltk.download('punkt')

# Utility functions and data preprocessing

def device_use():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def preprocess_text(text):
    """Preprocess the text by tokenizing, removing special characters, and converting to lowercase."""
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        sentence = sentence.lower()
        words = word_tokenize(sentence)
        if words:
            cleaned_sentences.append(words)
    return cleaned_sentences

def load_data(file_path):
    """Loads text from a file and returns the cleaned sentences."""
    with open(file_path, 'r') as f:
        text = f.read()
    return preprocess_text(text)

def build_vocab(sentences, glove_dictionary):
    """Builds the vocabulary from sentences based on pre-trained glove embeddings."""
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for sent in sentences:
        for word in sent:
            if word in glove_dictionary:
                if word not in vocab:
                    vocab[word] = len(vocab)
    return vocab

def load_glove_embeddings(glove_path):
    """Loads GloVe embeddings from a file."""
    glove_dict = {}
    with open(glove_path, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = [float(x) for x in values[1:]]
            glove_dict[word] = word_embedding
    return glove_dict

# Dataset definition

class TextDataset(Dataset):
    def __init__(self, sentences, vocab, embedding_matrix):
        self.sentences = sentences
        self.vocab = vocab
        self.embedding_matrix = embedding_matrix
        self.data = self.create_sequences()

    def create_sequences(self):
        """Creates input and output sequences for training."""
        sequences = []
        for sent in self.sentences:
            if len(sent) > 1:
                input_seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in sent[:-1]]
                output_seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in sent[1:]]
                if len(input_seq) > 0 and len(output_seq) > 0:
                    sequences.append((input_seq, output_seq))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, output_seq = self.data[idx]
        input_embeds = torch.tensor(input_seq, dtype=torch.long)
        output_seq = torch.tensor(output_seq, dtype=torch.long)
        return input_embeds, output_seq

# Transformer Model definition

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.5):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + self.norm1(attn_output)
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return x + self.norm2(x2)

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len, dropout=0.5, embedding_matrix=None):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
                                             for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        x = self.embedding(x) * np.sqrt(x.size(-1))
        x = self.positional_encoding(x.transpose(0, 1))
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask)
        return self.fc_out(x.transpose(0, 1))

# Utility functions for training and evaluation

def generate_square_subsequent_mask(sz):
    """Generates a square subsequent mask."""
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def collate_fn(batch, vocab):
    """Custom collate function to pad sequences."""
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, targets = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=vocab['<PAD>'])
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=vocab['<PAD>'])
    lengths = [len(seq) for seq in sequences]
    return sequences_padded, targets_padded, torch.tensor(lengths)

def train_epoch(model, data_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for input_batch, target_batch, lengths in data_loader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        tgt_mask = generate_square_subsequent_mask(input_batch.size(1)).to(device)
        optimizer.zero_grad()
        output = model(input_batch, tgt_mask)
        output = output.view(-1, output.size(-1))
        target_batch = target_batch.view(-1)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on validation/test dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_batch, target_batch, lengths in data_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            tgt_mask = generate_square_subsequent_mask(input_batch.size(1)).to(device)
            output = model(input_batch, tgt_mask)
            output = output.view(-1, output.size(-1))
            target_batch = target_batch.view(-1)
            loss = criterion(output, target_batch)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_and_save_perplexity(model, data_loader, criterion, device, file_name):
    """Evaluate model and save perplexity scores to a file."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with open(file_name, 'w') as f:
            for batch_idx, (input_batch, target_batch, lengths) in enumerate(data_loader):
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                tgt_mask = generate_square_subsequent_mask(input_batch.size(1)).to(device)
                output = model(input_batch, tgt_mask)
                output = output.view(-1, output.size(-1))
                target_batch = target_batch.view(-1)
                loss = criterion(output, target_batch)
                total_loss += loss.item()
                batch_perplexity = calculate_perplexity(loss.item())
                f.write(f'{batch_idx + 1}\t{batch_perplexity:.4f}\n')
            average_loss = total_loss / len(data_loader)
            average_perplexity = calculate_perplexity(average_loss)
            f.write(f'Average\t{average_perplexity:.4f}\n')

if __name__ == '__main__':
    # Load data and preprocess
    data_path = 'Data/Auguste_Maquet.txt'
    glove_path = 'Data/glove.6B.100d.txt'
    sentences = load_data(data_path)
    glove_dict = load_glove_embeddings(glove_path)
    random.shuffle(sentences)
    train_split = int(0.7 * len(sentences))
    test_split = int(0.9 * len(sentences))

    train_sentences = sentences[:train_split]
    test_sentences = sentences[train_split:test_split]
    val_sentences = sentences[test_split:]

    vocab = build_vocab(train_sentences, glove_dict)

    embedding_matrix = [glove_dict[word] if word in glove_dict else glove_dict['<UNK>'] for word in vocab]
    device = device_use()

    # Create dataset and dataloader

    train_dataset = TextDataset(train_sentences, vocab, embedding_matrix)
    val_dataset = TextDataset(val_sentences, vocab, embedding_matrix)
    test_dataset = TextDataset(test_sentences, vocab, embedding_matrix)

    BATCH_SIZE=64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=lambda x: collate_fn(x, vocab))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  collate_fn=lambda x: collate_fn(x, vocab))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, vocab))

    BATCH_SIZE_FILE=1
    train_loader_file = DataLoader(train_dataset, batch_size=BATCH_SIZE_FILE, shuffle=True,  collate_fn=lambda x: collate_fn(x, vocab))
    val_loader_file = DataLoader(val_dataset, batch_size=BATCH_SIZE_FILE, shuffle=False,  collate_fn=lambda x: collate_fn(x, vocab))
    test_loader_file = DataLoader(test_dataset, batch_size=BATCH_SIZE_FILE, shuffle=False,  collate_fn=lambda x: collate_fn(x, vocab))

    # Model configuration
    embedding_matrix = np.array(embedding_matrix)  # Ensure it's a numpy array
    model = TransformerLanguageModel(
        vocab_size=len(vocab),
        d_model=300,  # or the appropriate value
        nhead=10,
        num_layers=2,
        dim_feedforward=300,
        max_len=400,
        embedding_matrix=embedding_matrix
        ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Training Loop
    EPOCHS = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=3

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        val_perplexity = calculate_perplexity(val_loss)
        # Early stopping logic
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    test_loss = evaluate(model, test_loader, criterion, device)
    test_perplexity = calculate_perplexity(test_loss)
    print(f'Test Loss: {test_loss:.4f} ,Test Perplexity: {test_perplexity:.4f}')

    test_loss, test_perplexity = evaluate_and_save_perplexity(
    model, test_loader_file, criterion, device, '2021101075-LM3-test-perplexity.txt')

    val_loss, val_perplexity = evaluate_and_save_perplexity(
    model, val_loader_file, criterion, device, '2021101075-LM3-val-perplexity.txt')

    train_loss, train_perplexity = evaluate_and_save_perplexity(
    model, train_loader_file, criterion, device, '2021101075-LM3-train-perplexity.txt')

    print(f'Train Loss: {train_loss:.4f} , Train Perplexity: {train_perplexity:.4f}')
    print(f'val Loss: {val_loss:.4f} , val Perplexity: {val_perplexity:.4f}')
    print(f'test Loss: {test_loss:.4f} , Test Perplexity: {test_perplexity:.4f}')