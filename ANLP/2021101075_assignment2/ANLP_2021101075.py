
#@title IMPORT REQUIRED LIIBRARIES
import torch
import torch.nn as nn
import math
import re
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import csv
import os
from google.colab import files
from nltk.translate.bleu_score import SmoothingFunction

"""## TESTING HELPER FUNCTION"""

# Define references and hypotheses
references_1 = [['this', 'is', 'a', 'test']]  # Reference for the first hypothesis
hypothesis_1 = ['this', 'is', 'test']  # Hypothesis to evaluate

# Define a smoothing function
chencherry = SmoothingFunction()

# Calculate BLEU score for the first sentence with smoothing
score_1 = sentence_bleu(references_1, hypothesis_1, smoothing_function=chencherry.method1)
print(f"BLEU score for the first hypothesis (with smoothing): {score_1}")

# Define references and hypotheses
references_1 = [['this', 'is', 'a', 'test']]  # Reference for the first hypothesis
hypothesis_1 = ['this', 'is', 'test']  # Hypothesis to evaluate

# Calculate BLEU score for the first sentence
score_1 = sentence_bleu(references_1, hypothesis_1)
print(f"BLEU score for the first hypothesis: {score_1}")

# Define second reference and hypothesis
references_2 = [['another', 'sentence']]  # Reference for the second hypothesis
hypothesis_2 = ['another', 'example']  # Hypothesis to evaluate

# Calculate BLEU score for the second sentence
score_2 = sentence_bleu(references_2, hypothesis_2)
print(f"BLEU score for the second hypothesis: {score_2}")

# Average BLEU score for both hypotheses
average_score = (score_1 + score_2) / 2
print(f"Score 1 ",{score_1})
print(f"Score 2 ",{score_2})
print(f"Average BLEU score: {average_score}")

from nltk.translate.bleu_score import sentence_bleu

# Define references and hypotheses
references_1 = [['this', 'is', 'a', 'test']]  # Reference for the first hypothesis
hypothesis_1 = ['this', 'is', 'test']  # Hypothesis to evaluate

# Define weights for 1-gram, 2-gram, 3-gram, and 4-gram
weights = (0.8, 0.2, 0, 0)

# Calculate BLEU score for the first sentence with the defined weights
score_1 = sentence_bleu(references_1, hypothesis_1, weights=weights)
print(f"BLEU score for the first hypothesis: {score_1}")

# Define second reference and hypothesis
references_2 = [['another', 'sentence']]  # Reference for the second hypothesis
hypothesis_2 = ['another', 'example']  # Hypothesis to evaluate

# Calculate BLEU score for the second sentence with the defined weights
score_2 = sentence_bleu(references_2, hypothesis_2, weights=weights)
print(f"BLEU score for the second hypothesis: {score_2}")

# Average BLEU score for both hypotheses
average_score = (score_1 + score_2) / 2
print(f"Score 1: {score_1}")
print(f"Score 2: {score_2}")
print(f"Average BLEU score: {average_score}")

"""## TRANSFORMER ARCHITECTURE"""

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

        #Scaling Input Embeddings by the factor sqrt(d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position_value = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        divide_by = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        pe[:, 0::2] = torch.sin(position_value * divide_by) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position_value * divide_by) # cos(position * (10000 ** (2i / d_model))
        pe = pe.unsqueeze(0) # (1, seq_len, d_model) , including batch
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        x = self.dropout(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # multiply
        self.bias = nn.Parameter(torch.zeros(features)) # addition

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model->d_ff->d_model)
        x= self.linear_1(x)
        x = self.dropout(torch.relu(x))
        x = self.linear_2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "Embeddings can't be distributed in h segments."

        self.d_k = d_model // h   # vector size seen by h head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_model)*(d_model,seq_len) --> (batch, h, seq_len, seq_len) QKT/root(dk)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)


        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)n
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            # sublayer is the layer with whom you wan't to make residual connection with
            return x + self.dropout(sublayer(self.norm(x)))

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)   #here in the video it is return torch.log_softmax(self.proj(x),dim=-1)

"""**ENCODER**"""

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout),ResidualConnection(features,dropout)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

"""**DECODER**"""

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout),ResidualConnection(features,dropout),ResidualConnection(features,dropout)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        x= self.norm(x)
        return x

"""**TRANSFORMER**"""

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model, N, h, dropout, d_ff) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

"""**LOADING DATASET**"""

def read_datasets(inputfilename,outputfilename):
    ds = {}
    with open(inputfilename, 'r', encoding='utf-8') as file_en:
        ds['english'] = [line.strip() for line in file_en.readlines()]
    with open(outputfilename, 'r', encoding='utf-8') as file_fr:
        ds['french'] = [line.strip() for line in file_fr.readlines()]
    return ds

def calculate_statistics(ds, lang):
    if lang == 'en':
        sentences = ds['english']
    elif lang == 'fr':
        sentences = ds['french']

    lengths = [len(sentence.split()) for sentence in sentences]

    average_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0

    return average_length, min_length, max_length

def truncate_sentences(ds, lang, max_tokens=55):
    if lang == 'en':
        lines = ds['english']
    elif lang == 'fr':
        lines = ds['french']

    # Update the sentences in the dataset
    for i in range(len(lines)):
        tokenized_line = lines[i].split()
        if len(tokenized_line) > max_tokens:
            tokenized_line = tokenized_line[:max_tokens]
        lines[i] = ' '.join(tokenized_line).strip()

def clean_text(text):
    """Remove special characters from the text."""
    cleaned_text = re.sub(r'[^a-zA-Z0-9éèêëôîâä\s]', '', text)
    return cleaned_text.strip()


def clean_dataset(ds):
    """Remove special characters from sentences in the dataset."""
    ds['english'] = [clean_text(sentence) for sentence in ds['english']]
    ds['french'] = [clean_text(sentence) for sentence in ds['french']]
    return ds

train_input_sentences_file='train.en'
train_output_sentences_file='train.fr'

val_input_sentences_file='dev.en'
val_output_sentences_file='dev.fr'

test_input_sentences_file='test.en'
test_output_sentences_file='test.fr'


train_ds=read_datasets(train_input_sentences_file,train_output_sentences_file)
val_ds=read_datasets(val_input_sentences_file,val_output_sentences_file)
test_ds=read_datasets(test_input_sentences_file,test_output_sentences_file)

print(len(train_ds['english']))
print(len(train_ds['french']))

train_ds = clean_dataset(train_ds)
val_ds = clean_dataset(val_ds)
test_ds = clean_dataset(test_ds)

print(len(train_ds['english']))
print(len(train_ds['french']))

def get_or_build_tokenizer( ds, lang):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang,55), trainer=trainer)
    return tokenizer

def get_all_sentences(ds, lang, max_tokens=55):
    max_len = 0
    if lang == 'en':
        lines = ds['english']
    elif lang == 'fr':
        lines = ds['french']

    for line in lines:
        tokenized_line = line.split()
        if len(tokenized_line) > max_tokens:
            tokenized_line = tokenized_line[:max_tokens]
        max_len = max(max_len, len(tokenized_line))
        yield ' '.join(tokenized_line).strip()

tokenizer_src=get_or_build_tokenizer(train_ds, 'en')
tokenizer_tgt=get_or_build_tokenizer(train_ds, 'fr')

vocab_size_src=tokenizer_src.get_vocab_size()
vocab_size_tgt=tokenizer_tgt.get_vocab_size()
print(vocab_size_src)
print(vocab_size_tgt)

average_length_english, min_length_english, max_length_english = calculate_statistics(train_ds, 'en')
average_length_french, min_length_french, max_length_french = calculate_statistics(train_ds, 'fr')

print(f"Average length of English sentences: {average_length_english}")
print(f"Minimum length of English sentences: {min_length_english}")
print(f"Maximum length of English sentences: {max_length_english}")

print(f"Average length of French sentences: {average_length_french}")
print(f"Minimum length of French sentences: {min_length_french}")
print(f"Maximum length of French sentences: {max_length_french}")

truncate_sentences(train_ds, 'en',55)
truncate_sentences(train_ds, 'fr',55)

truncate_sentences(val_ds, 'en',55)
truncate_sentences(val_ds, 'fr',55)

truncate_sentences(test_ds, 'en',55)
truncate_sentences(test_ds, 'fr',55)

print(len(train_ds['english']))
print(len(train_ds['french']))

maximum_length_src = 0
maximum_length_tgt = 0

for en_sentence, fr_sentence in zip(val_ds['english'], val_ds['french']):
    src_ids = tokenizer_src.encode(en_sentence).ids
    tgt_ids = tokenizer_tgt.encode(fr_sentence).ids
    maximum_length_src = max(maximum_length_src, len(src_ids))
    maximum_length_tgt = max(maximum_length_tgt, len(tgt_ids))

print(maximum_length_src)
print(maximum_length_tgt)

maximum_sequence_length=max(maximum_length_src,maximum_length_tgt)
print(maximum_sequence_length)

"""**CREATING DATASET**"""

class BuildDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        tokens=[torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64),torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64),torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)]
        self.sos_token= tokens[0]
        self.eos_token= tokens[1]
        self.pad_token= tokens[2]

    def __len__(self):
        return len(self.ds[self.src_lang])

    def __getitem__(self, idx):
        src_text = self.ds[self.src_lang][idx]
        tgt_text = self.ds[self.tgt_lang][idx]


        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate the required padding
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # For <sos> and <eos>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # For <sos>


        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Create encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create label (with eos token)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Ensure the sizes are correct

        if(encoder_input.size(0)!=self.seq_len or decoder_input.size(0)!=self.seq_len or label.size(0)!=self.seq_len):
            raise ValueError("Length could not be made equal to sequence length")

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.ones((1, size, size))
    mask = mask.triu(diagonal=1).type(torch.int)
    return mask == 0

training_dataset = BuildDataset(train_ds, tokenizer_src, tokenizer_tgt, 'english', 'french', seq_len=65)
val_dataset = BuildDataset(val_ds, tokenizer_src, tokenizer_tgt, 'english', 'french', seq_len=65)
test_dataset= BuildDataset(test_ds, tokenizer_src, tokenizer_tgt, 'english', 'french', seq_len=65)

print(len(training_dataset))
print(len(train_ds['english']))
print(len(train_ds['french']))

config = {
    'batch_size': 32,
    'src_seq_len':65,
    'tgt_seq_len': 65,
    'vocab_src': tokenizer_src.get_vocab_size() ,
    'vocab_tgt': tokenizer_tgt.get_vocab_size(),
    'd_model': 300 ,
    'N': 2,
    'h': 4,
    'dropout': 0.1,
    'd_ff': 1024,
    'learning_rate': 0.0005,
    'num_epochs': 15,
    'eps':1e-9
}

train_dataloader= DataLoader(training_dataset,config['batch_size'], shuffle=True)
val_dataloader= DataLoader(val_dataset,config['batch_size'], shuffle=True)
test_dataloader= DataLoader(test_dataset,1, shuffle=True) # since we need sentence wise bleu score

model=build_transformer(config['vocab_src'], config['vocab_tgt'], config['src_seq_len'], config['tgt_seq_len'], config['d_model'], config['N'], config['h'], config['dropout'], config['d_ff'])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98),eps=config['eps'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, min_lr=1e-6)

loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)

def calculate_metrics(model, dataloader, tokenizer_tgt, device,loss_fn):
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        total_loss = 0
        for batch in tqdm(dataloader, desc="Calculating Metrics"):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Get predictions (argmax)
            predictions = torch.argmax(proj_output, dim=-1)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()

            # Convert to list of sentences (token IDs to words)
            for i in range(predictions.size(0)):
                pred_sentence = predictions[i].cpu().numpy()
                ref_sentence = label[i].cpu().numpy()

                # Remove padding (0s) from predictions and references
                pred_sentence = pred_sentence[pred_sentence != 0]
                ref_sentence = ref_sentence[ref_sentence != 0]

                # Decode sentences using the tokenizer
                pred_sentence = tokenizer_tgt.decode(pred_sentence)
                ref_sentence = tokenizer_tgt.decode(ref_sentence)

                all_predictions.append(pred_sentence)
                all_references.append(ref_sentence)

    # Compute BLEU score
    references = [[ref.split()] for ref in all_references]  # List of references for BLEU
    predictions = [pred.split() for pred in all_predictions]  # List of predictions for BLEU
    bleu_score = corpus_bleu(references, predictions,smoothing_function=SmoothingFunction().method1)

    # Compute ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, pred in zip(all_references, all_predictions):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure  # Accumulate scores

    # Average ROUGE scores
    num_samples = len(all_references)
    for key in rouge_scores:
        rouge_scores[key] /= num_samples

    return bleu_score,rouge_scores, total_loss / len(dataloader)

csv_file = 'training_log.csv'

# Function to initialize the CSV file and log the model configuration
def init_csv_logging(config, csv_file):
    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Configuration'])
        for key, value in config.items():
            writer.writerow([key, value])
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'BLEU Score', 'ROUGE1', 'ROUGE2', 'ROUGE-L'])

def log_metrics(epoch, train_loss, val_loss, bleu_score, rouge_scores, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            bleu_score,
            rouge_scores['rouge1'],
            rouge_scores['rouge2'],
            rouge_scores['rougeL']
        ])

init_csv_logging(config, csv_file)

best_val_loss = float('inf')
patience_counter = 0
patience = 2

for epoch in range(config['num_epochs']):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch +1}')

    total_loss = 0  # Initialize total loss for the epoch
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device)  # (batchsize, seq_len)
        decoder_input = batch['decoder_input'].to(device)  # (batchsize, seq_len)
        encoder_mask = batch['encoder_mask'].to(device)    # (batchsize, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'].to(device)    # (batchsize, 1, seq_len, seq_len)
        label = batch['label'].to(device)                  # (batchsize, seq_len)

        # Forward pass
        encoder_output = model.encode(encoder_input, encoder_mask)  # (batchsize, seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (batchsize, seq_len, d_model)
        proj_output = model.project(decoder_output)  # (batchsize, seq_len, vocab_size)

        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

        total_loss += loss.item()
        batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update optimizer
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_train_loss:.4f}")

    # Validate the model
    with torch.no_grad():
        bleu_score,rouge_scores_val, val_loss = calculate_metrics(model, val_dataloader, tokenizer_tgt, device, loss_fn)

    # scheduler.step(val_loss)

    # Print the validation results
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], "
          f"Validation Loss: {val_loss:.4f}, ")

    # Log metrics to the CSV file (separate columns for rouge1, rouge2, and rougeL)
    log_metrics(epoch + 1, avg_train_loss, val_loss, bleu_score, rouge_scores_val, csv_file)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(torch.load('best_model.pt'))

def calculate_metrics_print(model, dataloader, tokenizer_tgt, device, loss_fn, output_file='testbleu.txt'):
    model.eval()
    all_predictions = []
    all_references = []
    num_samples = 0  # To keep track of the number of samples

    with torch.no_grad():
        total_loss = 0
        for batch in tqdm(dataloader, desc="Calculating Metrics"):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Get predictions (argmax)
            predictions = torch.argmax(proj_output, dim=-1)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()

            # Convert to list of sentences (token IDs to words)
            for i in range(predictions.size(0)):
                pred_sentence = predictions[i].cpu().numpy()
                ref_sentence = label[i].cpu().numpy()

                # Remove padding (0s) from predictions and references
                pred_sentence = pred_sentence[pred_sentence != 0]
                ref_sentence = ref_sentence[ref_sentence != 0]

                # Decode sentences using the tokenizer
                pred_sentence_decoded = tokenizer_tgt.decode(pred_sentence)
                ref_sentence_decoded = tokenizer_tgt.decode(ref_sentence)

                all_predictions.append(pred_sentence_decoded)
                all_references.append(ref_sentence_decoded)
                num_samples += 1  # Increment sample count


    # Open the output file for writing scores
    total_bleu_score=0
    with open(output_file, 'w') as f:  # Open the output file
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Compute BLEU and ROUGE scores for each sentence
        for idx, (ref, pred) in enumerate(zip(all_references, all_predictions), start=1):
            # Calculate BLEU score for the current prediction
            reference_tokens = [ref.split()]  # BLEU expects a list of reference sentences
            prediction_tokens = pred.split()
            sentence_bleu_score = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=SmoothingFunction().method1)
            total_bleu_score+=sentence_bleu_score

            # Calculate ROUGE scores
            scores = scorer.score(ref, pred)
            rouge1_score = scores['rouge1'].fmeasure
            rouge2_score = scores['rouge2'].fmeasure
            rougeL_score = scores['rougeL'].fmeasure

            # Write scores to the file
            f.write(f"{idx}\t{sentence_bleu_score}\t{rouge1_score:.4f}\t{rouge2_score:.4f}\t{rougeL_score:.4f}\n")


    average_bleu_score = total_bleu_score / num_samples
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, pred in zip(all_references, all_predictions):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure  # Accumulate scores

    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= num_samples

    # Append overall scores to the output file
    with open(output_file, 'a') as f:  # Append to the same file
        f.write(f"\nOverall Corpus BLEU Score: {average_bleu_score:.4f}\n")
        f.write(f"Average ROUGE1 Score: {rouge_scores['rouge1']:.4f}\n")
        f.write(f"Average ROUGE2 Score: {rouge_scores['rouge2']:.4f}\n")
        f.write(f"Average ROUGEL Score: {rouge_scores['rougeL']:.4f}\n")

    return average_bleu_score, rouge_scores, total_loss / len(dataloader)

bleu_score, rouge_scores, test_loss = calculate_metrics_print(model, test_dataloader, tokenizer_tgt, device, loss_fn, output_file='testbleu.txt')

print("test_bleu_score",bleu_score)
print("test_rouge_scores",rouge_scores)
print("test_loss",test_loss)

train_bleu_score, train_rouge_scores, train_loss = calculate_metrics_print(model, train_dataloader, tokenizer_tgt, device, loss_fn, output_file='trainbleu.txt')

print("train_bleu_score",train_bleu_score)
print("train_rouge_scores",train_rouge_scores)
print("train_loss",train_loss)

