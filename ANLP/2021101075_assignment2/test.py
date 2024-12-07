import torch
from utils import test_dataloader,calculate_metrics_print,tokenizer_tgt,device,loss_fn,Transformer,build_transformer,config,train_dataloader

if __name__ == "__main__":
    model=build_transformer(config['vocab_src'], config['vocab_tgt'], config['src_seq_len'], config['tgt_seq_len'], config['d_model'], config['N'], config['h'], config['dropout'], config['d_ff'])
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode    
    bleu_score, rouge_scores, test_loss = calculate_metrics_print(model, test_dataloader, tokenizer_tgt, device, loss_fn, output_file='testbleu.txt')
    print(f"Test BLEU Score: {bleu_score}")
    print(f"Test Rouge Scores: {rouge_scores}")
    print(f"Test Loss: {test_loss}")






