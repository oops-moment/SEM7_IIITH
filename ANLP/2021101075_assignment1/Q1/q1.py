import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import wandb

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize WandB inorder to log the result of hyper parameter tuning
wandb.init(project="ANLP Hyperparameter Tuning")

# Text preprocessing
def preprocess_text(text):
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence).lower()
        words = word_tokenize(sentence)
        if words:
            cleaned_sentences.append(words)
    return cleaned_sentences

def load_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return preprocess_text(text)

# Build vocab
def build_vocab(sentences, glove_dict):
    vocab = {'<UNK>': 0}
    for sent in sentences:
        for word in sent:
            if word in glove_dict and word not in vocab:
                vocab[word] = len(vocab)
    return vocab

# Load Glove embeddings
def load_glove(file_path):
    glove_dict = {}
    with open(file_path, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = [float(x) for x in values[1:]]
            glove_dict[word] = word_embedding
    return glove_dict

# Build n-grams
def build_ngrams(sentences, vocab, n=5):
    inputs, labels = [], []
    for sent in sentences:
        if len(sent) < n + 1:
            continue
        for i in range(len(sent) - n):
            prefix = sent[i:i + n]
            label = sent[i + n]
            encoded_prefix = [vocab.get(word, vocab['<UNK>']) for word in prefix]
            encoded_label = vocab.get(label, vocab['<UNK>'])
            inputs.append(encoded_prefix)
            labels.append(encoded_label)
    return inputs, labels

# NLM model
class NLM(nn.Module):
    def __init__(self, d_embedding, d_hidden, len_vocab, embedding_matrix, n=5):
        super(NLM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False).to(device)
        self.fc1 = nn.Linear(d_embedding * n, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, len_vocab)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.embeddings(x)
        x = x.view(x.size(0), -1)
        x = self.gelu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Train function
def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, early_stop_patience=2):
    best_val_loss = float('inf')
    count_early_stopping = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, loss_fn, val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            count_early_stopping = 0
        else:
            count_early_stopping += 1
            if count_early_stopping == early_stop_patience:
                print("Early stopping triggered.")
                break
    model.load_state_dict(torch.load('best_model.pth'))

# Evaluation function
def evaluate(model, loss_fn, data_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(data_loader)

# Perplexity calculation
def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# Hyperparameter tuning function
def train_and_evaluate(optimizer_type, d_hidden, dropout_rate, train_loader, val_loader, test_loader, num_epochs=5):
    model = NLM(d_embedding=300, d_hidden=d_hidden, len_vocab=len(vocab), embedding_matrix=embedding_matrix, n=5).to(device)
    model.dropout = nn.Dropout(dropout_rate)  # already on device

    optimizer = optim.Adam(model.parameters(), lr=0.01) if optimizer_type == 'adam' else optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Initialize or reset WandB
    wandb.init(project="language_model", config={
        "optimizer": optimizer_type,
        "d_hidden": d_hidden,
        "dropout_rate": dropout_rate,
        "learning_rate": 0.01,
        "epochs": num_epochs,
        "n_gram_size": 5
    })
    wandb.watch(model, log="all", log_freq=10)

    train(model, optimizer, nn.CrossEntropyLoss().to(device), train_loader, val_loader, num_epochs)
    
    val_perplexity = calculate_perplexity(model, val_loader)
    print(f"Validation Perplexity: {val_perplexity}")
    wandb.log({"val_perplexity": val_perplexity})
    
    test_perplexity = calculate_perplexity(model, test_loader)
    print(f"Test Perplexity: {test_perplexity}")
    wandb.log({"test_perplexity": test_perplexity})
    
    return val_perplexity, test_perplexity


def visualize_predictions(model, data_inputs, data_labels, index_to_word, num_matches=20, num_mismatches=10):
    model.eval()
    matches_visualized = 0
    mismatches_visualized = 0
    total_visualized = 0

    with torch.no_grad():
        for inputs, labels in zip(data_inputs, data_labels):
            if matches_visualized >= num_matches and mismatches_visualized >= num_mismatches:
                break

            inputs = inputs.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            outputs = model(inputs)
            _, predicted_label_index = torch.max(outputs, 1)
            predicted_label_index = predicted_label_index.item()
            true_label_index = labels.item()

            input_words = [index_to_word[idx] for idx in inputs.squeeze(0).cpu().numpy()]
            true_label_word = index_to_word.get(true_label_index, '<UNK>')
            predicted_label_word = index_to_word.get(predicted_label_index, '<UNK>')

            # Check if the prediction matches the true label
            is_match = (predicted_label_index == true_label_index)

            if (is_match and matches_visualized < num_matches) or (not is_match and mismatches_visualized < num_mismatches):
                # Print the results
                print(f"Input context: {' '.join(input_words)}")
                print(f"True next word: {true_label_word}")
                print(f"Predicted next word: {predicted_label_word}")
                print("-" * 50)

                # Update counters
                if is_match:
                    matches_visualized += 1
                else:
                    mismatches_visualized += 1
                total_visualized += 1

            if total_visualized >= (num_matches + num_mismatches):
                break

# Main flow
if __name__ == "__main__":
    file_path = 'Data/Auguste_Maquet.txt'
    glove_path = 'Data/glove.6B.300d.txt'
    sentences = load_data(file_path)
    glove_dict = load_glove(glove_path)

    random.shuffle(sentences)
    train_split = int(0.7 * len(sentences))
    test_split = int(0.9 * len(sentences))

    train_sentences = sentences[:train_split]
    test_sentences = sentences[train_split:test_split]
    val_sentences = sentences[test_split:]

    vocab = build_vocab(train_sentences, glove_dict)
    embedding_matrix = [glove_dict[word] if word in glove_dict else [0]*300 for word in vocab]

    train_inputs, train_labels = build_ngrams(train_sentences, vocab, n=5)
    test_inputs, test_labels = build_ngrams(test_sentences, vocab, n=5)
    val_inputs, val_labels = build_ngrams(val_sentences, vocab, n=5)

    train_inputs = torch.LongTensor(train_inputs).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)
    val_inputs = torch.LongTensor(val_inputs).to(device)
    val_labels = torch.LongTensor(val_labels).to(device)
    test_inputs = torch.LongTensor(test_inputs).to(device)
    test_labels = torch.LongTensor(test_labels).to(device)

    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)


    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model Parameters
    d_embedding = 300
    d_hidden = 300
    len_vocab = len(vocab)  # Assuming vocab is pre-built
    n_gram_size = 5  # Define the n-gram size
    learning_rate = 0.01
    num_epochs = 10

    # Initialize the model, loss function, and optimizer
    model = NLM(d_embedding, d_hidden, len_vocab, embedding_matrix, n=n_gram_size).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    # Initialize optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model,optimizer,loss_fn,train_loader,val_loader,num_epochs,3)

    # evalute the model
    test_loss=evaluate(model,loss_fn,test_loader)
    test_perplexity=calculate_perplexity(model,test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}")
    index_to_word = {index: word for word, index in vocab.items()}
    index = 204
    word = index_to_word.get(index, '<UNK>')

    visualize_predictions(model, test_inputs, test_labels, index_to_word, num_matches=20, num_mismatches=10)
    
    # Analyze optimizer first
    best_optimizer = None
    best_optimizer_perplexity = float('inf')
    optimizers = ['adam', 'sgd']
    optimizers_testppl=[]
    optimizers_valppl=[]
    for opt in optimizers:
        print(f"Testing optimizer: {opt}")
        val_ppl, test_ppl= train_and_evaluate(opt, 300, 0.5,train_loader,val_loader,test_loader)
        optimizers_testppl.append(test_ppl)
        optimizers_valppl.append(val_ppl)
        if test_ppl < best_optimizer_perplexity:
            best_optimizer = opt
            best_optimizer_perplexity = test_ppl
    
    print(f"Best optimizer: {best_optimizer}, Test Perplexity: {best_optimizer_perplexity}")

    # Analyze hidden dimension
    best_hidden_dim = None
    best_hidden_dim_perplexity = float('inf')
    hidden_dims = [200, 300, 400]
    hidden_dims_testppl=[]
    hidden_dims_valppl=[]

    for hidden_dim in hidden_dims:
        print(f"Testing hidden dimension: {hidden_dim}")
        val_ppl, test_ppl = train_and_evaluate(best_optimizer, hidden_dim, 0.5,train_loader,val_loader,test_loader)
        hidden_dims_testppl.append(test_ppl)
        hidden_dims_valppl.append(val_ppl)
        if test_ppl < best_hidden_dim_perplexity:
            best_hidden_dim = hidden_dim
            best_hidden_dim_perplexity = test_ppl

    print(f"Best hidden dimension: {best_hidden_dim}, Test Perplexity: {best_hidden_dim_perplexity}")


    # Analyze dropout rate

    best_dropout_rate = None
    best_dropout_rate_perplexity = float('inf')
    dropout_rates = [0.3, 0.5, 0.7]
    dropout_rates_testppl=[]
    dropout_rates_valppl=[]

    for dropout_rate in dropout_rates:
        print(f"Testing dropout rate: {dropout_rate}")
        val_ppl, test_ppl= train_and_evaluate(best_optimizer, best_hidden_dim, dropout_rate,train_loader,val_loader,test_loader)
        dropout_rates_testppl.append(test_ppl)
        dropout_rates_valppl.append(val_ppl)
        if test_ppl < best_dropout_rate_perplexity:
            best_dropout_rate = dropout_rate
            best_dropout_rate_perplexity = test_ppl
    
    print(f"Best dropout rate: {best_dropout_rate}, Test Perplexity: {best_dropout_rate_perplexity}")


    # Doing hyper parameter tuning

    for optimizer in optimizers:
        for hidden_dim in hidden_dims:
            for dropout_rate in dropout_rates:
                print(f"Testing optimizer: {optimizer}, hidden_dim: {hidden_dim}, dropout_rate: {dropout_rate}")
                train_and_evaluate(optimizer, hidden_dim, dropout_rate,train_loader,val_loader,test_loader)