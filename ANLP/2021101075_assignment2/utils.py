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
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import csv
import os
from nltk.translate.bleu_score import SmoothingFunction

from encoder import Encoder, EncoderBlock , MultiHeadAttentionBlock, FeedForwardBlock , LayerNormalization , ResidualConnection
from decoder import Decoder, DecoderBlock



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
        
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)   #here in the video it is return torch.log_softmax(self.proj(x),dim=-1)
    
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
    
    return ds

def clean_text(text):
    """Remove special characters from the text."""
    cleaned_text = re.sub(r'[^a-zA-Z0-9éèêëôîâä\s]', '', text)
    return cleaned_text.strip()


def clean_dataset(ds):
    """Remove special characters from sentences in the dataset."""
    ds['english'] = [clean_text(sentence) for sentence in ds['english']]
    ds['french'] = [clean_text(sentence) for sentence in ds['french']]
    return ds

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

def get_or_build_tokenizer( ds, lang):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang,55), trainer=trainer)
    return tokenizer

train_input_sentences_file='train.en'
train_output_sentences_file='train.fr'

val_input_sentences_file='dev.en'
val_output_sentences_file='dev.fr'

test_input_sentences_file='test.en'
test_output_sentences_file='test.fr'


train_ds=read_datasets(train_input_sentences_file,train_output_sentences_file)
val_ds=read_datasets(val_input_sentences_file,val_output_sentences_file)
test_ds=read_datasets(test_input_sentences_file,test_output_sentences_file)
train_ds = clean_dataset(train_ds)
val_ds = clean_dataset(val_ds)
test_ds = clean_dataset(test_ds)
tokenizer_src=get_or_build_tokenizer(train_ds, 'en')
tokenizer_tgt=get_or_build_tokenizer(train_ds, 'fr')

train_ds=   truncate_sentences(train_ds, 'en',55)
train_ds=   truncate_sentences(train_ds, 'fr',55)

val_ds=   truncate_sentences(val_ds, 'en',55)
val_ds=   truncate_sentences(val_ds, 'fr',55)

test_ds=   truncate_sentences(test_ds, 'en',55)
test_ds=   truncate_sentences(test_ds, 'fr',55)


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

    
training_dataset = BuildDataset(train_ds, tokenizer_src, tokenizer_tgt, 'english', 'french', seq_len=65)
val_dataset = BuildDataset(val_ds, tokenizer_src, tokenizer_tgt, 'english', 'french', seq_len=65)
test_dataset= BuildDataset(test_ds, tokenizer_src, tokenizer_tgt, 'english', 'french', seq_len=65)

train_dataloader= DataLoader(training_dataset,config['batch_size'], shuffle=True)
val_dataloader= DataLoader(val_dataset,config['batch_size'], shuffle=True)
test_dataloader= DataLoader(test_dataset,1, shuffle=True) # since we need sentence wise bleu score

model=build_transformer(config['vocab_src'], config['vocab_tgt'], config['src_seq_len'], config['tgt_seq_len'], config['d_model'], config['N'], config['h'], config['dropout'], config['d_ff'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98),eps=config['eps'])
loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)

csv_file = 'training_log.csv'

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