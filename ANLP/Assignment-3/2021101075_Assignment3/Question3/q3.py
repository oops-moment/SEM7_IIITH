# -*- coding: utf-8 -*-
"""Q3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PSh0zWBcjFpqCUUos2ydRiLR-bHyzJN3
"""
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from transformers import DataCollatorForLanguageModeling
import transformers
import time
from rouge_score import rouge_scorer  # Using rouge_score directly

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
foundation_model = AutoModelForCausalLM.from_pretrained(model_name)

# Freeze all model parameters except the language modeling (LM) head
for param in foundation_model.parameters():
    param.requires_grad = False
for param in foundation_model.lm_head.parameters():
    param.requires_grad = True

foundation_model

# Check the number of parameters to verify
total_params = sum(p.numel() for p in foundation_model.parameters())
trainable_params = sum(p.numel() for p in foundation_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

"""## Load the Dataset"""

# Load and preprocess the dataset
data = load_dataset("cnn_dailymail", "3.0.0")
tokenizer.pad_token = tokenizer.eos_token

# Using only a 10% of the data
train_size, test_size, validation_size = [int(0.1 * len(data[split])) for split in ['train', 'test', 'validation']]
train_sample = data['train'].select(range(train_size))
test_sample = data['test'].select(range(test_size))
validation_sample = data['validation'].select(range(validation_size))

# Preprocessing function
def preprocess_function(examples):
    model_inputs = tokenizer(examples["article"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data_train = train_sample.map(preprocess_function, batched=True)
tokenized_data_test = test_sample.map(preprocess_function, batched=True)
tokenized_data_validation = validation_sample.map(preprocess_function, batched=True)

# Define output directory for the fine-tuned model
output_directory = "./gpt2_finetuned_last_layers"

# Initialize ROUGE scorer
rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Compute added parameters
added_params = sum(p.numel() for p in foundation_model.lm_head.parameters() if p.requires_grad)

# Track computation time
start_time = time.time()

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_directory,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=50,
    save_steps=100,
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
    no_cuda=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Accumulate scores for each metric
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge_scorer_instance.score(pred, label)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    # Calculate mean scores for each ROUGE metric
    result = {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) * 100,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) * 100,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) * 100
    }
    return result

trainer = Trainer(
    model=foundation_model,
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_validation,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Start training
trainer.train()

# Track training end time
end_time = time.time()
total_time = end_time - start_time

print("Training time (seconds):", total_time)
print("Added parameters:", added_params)

# Evaluate the model
test_results = trainer.evaluate(tokenized_data_test)
print("Test results:", test_results)

# prompt: correct syntax PYTORCH_CUDA_ALLOC_CONF=expandable_segments:
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments"

testing_input_text=tokenized_data_test[0]["article"]
testing_output_text=tokenized_data_test[0]["highlights"]

print(testing_input_text)
print(testing_output_text)

testing_input_text=tokenized_data_test[0]["article"]
testing_output_text=tokenized_data_test[0]["highlights"]

model=foundation_model
model.eval()

# Adjust the generate_summary function to handle device compatibility and output length
def generate_summary(text, max_new_tokens=128):
    model.config.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    tokenizer.pad_token = tokenizer.eos_token
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,  # Controls only the generated text length
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(output[0][:max_new_tokens], skip_special_tokens=True)

# Generate summary for the test input
generated_summary = generate_summary(testing_input_text)
print("Generated Summary:", generated_summary)

# Calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(generated_summary, testing_output_text)

# Print ROUGE scores as percentages
print("ROUGE-1 Score:", scores["rouge1"].fmeasure )
print("ROUGE-2 Score:", scores["rouge2"].fmeasure )
print("ROUGE-L Score:", scores["rougeL"].fmeasure )


from tqdm import tqdm  # Import tqdm for progress bar

# Initialize lists to store ROUGE scores for all test samples
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

# Loop through the test dataset with progress tracking
for test_example in tqdm(tokenized_data_test, desc="Processing test examples", unit="example"):
    testing_input_text = tokenizer.decode(test_example["input_ids"], skip_special_tokens=True)
    testing_output_text = tokenizer.decode(test_example["labels"], skip_special_tokens=True)

    # Generate summary for each test input
    generated_summary = generate_summary(testing_input_text)

    # Calculate ROUGE scores for the generated summary against the reference summary
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(generated_summary, testing_output_text)

    # Append each score to the respective list
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

# Calculate the average ROUGE scores over the entire test dataset
average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

# Print the average ROUGE scores
print("Average ROUGE-1 Score:", average_rouge1)
print("Average ROUGE-2 Score:", average_rouge2)
print("Average ROUGE-L Score:", average_rougeL)

# Number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# Estimate GPU compute (in FLOPs)
flops = 2 * num_params * (training_args.per_device_train_batch_size * training_args.num_train_epochs)
print(f"Estimated FLOPs: {flops}")

# GPU memory usage
gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
print(f"GPU memory allocated: {gpu_memory:.2f} MB")

import matplotlib.pyplot as plt

steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]

training_loss = [3.350700, 3.402100, 3.361300, 3.338800, 3.336400, 3.250900, 3.258600, 3.247400, 3.208800, 3.236700, 3.183300, 3.228500, 3.244900, 3.182800, 3.180700, 3.155000, 3.181600, 3.184300, 3.217900, 3.145400, 3.166800, 3.185600, 3.189400, 3.159200]

validation_loss = [3.431694, 3.336799, 3.283325, 3.252369, 3.230885, 3.216898, 3.208629, 3.199492, 3.197267, 3.190312, 3.185227, 3.184411, 3.180120, 3.177222, 3.176097, 3.175769, 3.171443, 3.174385, 3.169318, 3.166926, 3.170565, 3.166215, 3.169252, 3.166747]

# Plot the training and validation loss
plt.plot(steps, training_loss, label="Training Loss")
plt.plot(steps, validation_loss, label="Validation Loss")

# Set the plot title and labels
plt.title("Training and Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")

# Add a legend
plt.legend()

# Display the plot
plt.show()
