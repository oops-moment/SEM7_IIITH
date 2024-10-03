import pandas as pd
import matplotlib.pyplot as plt

csv_file = './try2/training_log-6.csv' 

data = pd.read_csv(csv_file, skiprows=14)

config_params = {
    "batch_size": 32,
    "d_model": 512,
    "N": 6,
    "h": 8,
    "dropout": 0.1,
    "d_ff": 2048,
    "learning_rate": 0.001,
}

def plot_with_config(ax, x, y, title, xlabel, ylabel):
    ax.plot(x, y, marker='o')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

fig, ax = plt.subplots(figsize=(8, 5))
title = f'Validation Loss vs Epoch\n' + '\n'.join([f'{key}: {value}' for key, value in config_params.items()])
plot_with_config(ax, data['Epoch'], data['Validation Loss'], title, 'Epoch', 'Validation Loss')
plt.show()  # Show the first plot

# Create the second plot: BLEU Score vs Epoch
fig, ax = plt.subplots(figsize=(8, 5))
title = f'BLEU Score vs Epoch\n' + '\n'.join([f'{key}: {value}' for key, value in config_params.items()])
plot_with_config(ax, data['Epoch'], data['BLEU Score'], title, 'Epoch', 'BLEU Score')
plt.show()  # Show the second plot

# Create the third plot: ROUGE Scores vs Epoch
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(data['Epoch'], data['ROUGE1'], marker='o', label='ROUGE1')
ax.plot(data['Epoch'], data['ROUGE2'], marker='o', label='ROUGE2')
ax.plot(data['Epoch'], data['ROUGE-L'], marker='o', label='ROUGE-L')
title = f'ROUGE Scores vs Epoch\n' + '\n'.join([f'{key}: {value}' for key, value in config_params.items()])
ax.set_title(title)
ax.set_xlabel('Epoch')
ax.set_ylabel('ROUGE Score')
ax.grid(True)
ax.legend()

plt.show()  