# -*- coding: utf-8 -*-
"""q2_anlp.py"""

# Import Required Libraries
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

nltk.download('punkt')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Preprocessing Function
def preprocess_text(text):
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        sentence = sentence.lower()
        words = word_tokenize(sentence)
        if words:
            cleaned_sentences.append(words)
    return cleaned_sentences

# Load Data Function
def load_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return preprocess_text(text)

# Build Vocabulary
def build_vocab(sentences, glove_dictionary):
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for sent in sentences:
        for word in sent:
            if word in glove_dictionary and word not in vocab:
                vocab[word] = len(vocab)
    return vocab

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, sentences, vocab, embedding_matrix):
        self.sentences = sentences
        self.vocab = vocab
        self.embedding_matrix = embedding_matrix
        self.data = self.create_sequences()

    def create_sequences(self):
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

# Collate Function
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, targets = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=vocab['<PAD>'])
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=vocab['<PAD>'])
    lengths = [len(seq) for seq in sequences]
    return sequences_padded, targets_padded, torch.tensor(lengths)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_seq, lengths, hidden_state=None):
        embedded = self.dropout(self.embedding(input_seq))
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        if hidden_state is None:
            h_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_dim).to(input_seq.device)
            c_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_dim).to(input_seq.device)
            hidden_state = (h_0, c_0)

        packed_output, hidden_state = self.lstm(packed_input, hidden_state)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(lstm_out)
        return output, hidden_state

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h_0, c_0

# Train Model Function
def train_model(model, train_loader, val_loader, vocab, device, num_epochs=10, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_seq, target_seq, lengths in train_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            hidden = model.init_hidden(input_seq.size(0), device)
            optimizer.zero_grad()
            output, hidden = model(input_seq, lengths, hidden)
            loss = loss_fn(output.view(-1, len(vocab)), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq, lengths in val_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                output, _ = model(input_seq, lengths)
                loss = loss_fn(output.view(-1, len(vocab)), target_seq.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

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

# Calculate Perplexity Function
def calculate_perplexity(model, data_loader, vocab, device):
    model.eval()
    data_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    with torch.no_grad():
        for input_seq, target_seq, lengths in data_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output, _ = model(input_seq, lengths)
            loss = loss_fn(output.view(-1, len(vocab)), target_seq.view(-1))
            data_loss += loss.item()

    data_loss /= len(data_loader)
    perplexity = torch.exp(torch.tensor(data_loss))
    return perplexity, data_loss

def calculate_perplexity_and_save(model, data_loader, vocab, device, file_name):
    model.eval()
    data_loss = 0
    vocab_size = len(vocab)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    batch_losses = []

    with torch.no_grad():
        for batch_idx, (input_seq, target_seq, lengths) in enumerate(data_loader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            output, _ = model(input_seq, lengths)

            loss = loss_fn(output.view(-1, vocab_size), target_seq.view(-1))
            batch_loss = loss.item()
            batch_losses.append(batch_loss)

            data_loss += batch_loss

            batch_perplexity = torch.exp(torch.tensor(batch_loss)).item()

            # Save the perplexity score for this batch to the file
            with open(file_name, 'a') as f:
                f.write(f'Batch-{batch_idx + 1}\t{batch_perplexity}\n')

    avg_loss = data_loss / len(data_loader)
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    with open(file_name, 'a') as f:
        f.write(f'Average\t{avg_perplexity}\n')

    return avg_perplexity, avg_loss


# Main execution
if __name__ == '__main__':
    # Load and preprocess data
    
    file_path = 'Data/Auguste_Maquet.txt'
    sentences = load_data(file_path)

    # Load GloVe embeddings
    glove_dict = {}
    with open('Data/glove.6B.300d.txt', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = [float(x) for x in values[1:]]
            glove_dict[word] = word_embedding

    # Split data into train, test, and validation sets
    random.shuffle(sentences)
    train_split = int(0.7 * len(sentences))
    test_split = int(0.9 * len(sentences))
    train_sentences = sentences[:train_split]
    test_sentences = sentences[train_split:test_split]
    val_sentences = sentences[test_split:]

    vocab = build_vocab(train_sentences, glove_dict)

    embedding_matrix = [glove_dict[word] if word in glove_dict else [0]*300 for word in vocab]
    embedding_matrix = np.array(embedding_matrix)

    # Create datasets and loaders
    train_dataset = TextDataset(train_sentences, vocab, embedding_matrix)
    val_dataset = TextDataset(val_sentences, vocab, embedding_matrix)
    test_dataset = TextDataset(test_sentences, vocab, embedding_matrix)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = LSTMModel(embedding_matrix, hidden_dim=300, num_layers=2, dropout=0.5).to(device)

    # Train the model
    train_model(model, train_loader, val_loader, vocab, device)

    # Load the best model and calculate perplexity
    model.load_state_dict(torch.load('best_model.pth'))
    perplexity, avg_loss = calculate_perplexity(model, test_loader, vocab, device)
    print(f'Perplexity: {perplexity.item()}, Average Test Loss: {avg_loss:.4f}')

    BATCH_SIZE_FILE=1
    train_loader_file = DataLoader(train_dataset, batch_size=BATCH_SIZE_FILE, shuffle=True, collate_fn=collate_fn)
    val_loader_file = DataLoader(val_dataset, batch_size=BATCH_SIZE_FILE, shuffle=False, collate_fn=collate_fn)
    test_loader_file = DataLoader(test_dataset, batch_size=BATCH_SIZE_FILE, shuffle=False, collate_fn=collate_fn)

    calculate_perplexity_and_save(model, train_loader_file, vocab, device, '2021101075-LM2-train-perplexity.txt')
    calculate_perplexity_and_save(model, test_loader_file, vocab, device, '2021101075-LM2-test-perplexity.txt')
    calculate_perplexity_and_save(model, val_loader_file, vocab, device, '2021101075-LM2-val-perplexity.txt')
