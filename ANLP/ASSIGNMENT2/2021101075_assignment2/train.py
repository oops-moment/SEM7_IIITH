
import torch
from tqdm import tqdm

from utils import config,model,train_dataloader,device,loss_fn,tokenizer_src,tokenizer_tgt,optimizer,train_dataloader,val_dataloader,csv_file,calculate_metrics,log_metrics

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