# Assignment 1

**Advanced NLP Assignment 1**  
**Prisha 2021101075**

## Data Preparation

Before running the scripts, download the following datasets and models:
- **Models**: [Download Link](https://drive.google.com/drive/folders/1V3xt_IGs30uliz-RHXOnJCYJ7E2qlzbe?usp=sharing)
- **Text Corpus**: [Download Link](https://drive.google.com/file/d/1tZhhNsj0bepYWyyFXt7ZyD2F5aEwSkNS/view?usp=sharing)
- **GloVe Embedding File**: [Download Link](https://drive.google.com/file/d/1xQew23Eo-WxCe95bptox8GKDgFi0kJWT/view?usp=sharing)

Place these files in the `Data/` directory, as the code requires the data to be located there.

## Neural Network Language Model (NNLM) - (Q1)

- **`q1.py`**: Main script for:
  - Training a new N-gram model.

  **Usage:**
  - **Train from Scratch**: `python3 q1.py`
    - This script will preprocess the data, train the N-gram model, and save perplexity scores for different datasets.

- **`Q1.ipynb`**: Jupyter notebook detailing the entire workflow, including preprocessing, model training, and hyperparameter tuning.

- **`2021101075_LM1_test_perplexity.txt`**: Contains perplexity scores for the test dataset.
- **`2021101075_LM1_train_perplexity.txt`**: Contains perplexity scores for the training dataset.
- **`2021101075_LM1_val_perplexity.txt`**: Contains perplexity scores for the validation dataset.

## LSTM Model

- **`q2.py`**: Main script for:
  - Training a new LSTM model.

  **Usage:**
  - **Train from Scratch**: `python3 q2.py`
    - This script will preprocess the data, train the LSTM model, and save perplexity scores for different datasets.

- **`Q2.ipynb`**: Jupyter notebook detailing the entire workflow, including preprocessing, model training, and hyperparameter tuning for the LSTM model.

- **`2021101075_LM2_test_perplexity.txt`**: Contains perplexity scores for the test dataset.
- **`2021101075_LM2_train_perplexity.txt`**: Contains perplexity scores for the training dataset.
- **`2021101075_LM2_val_perplexity.txt`**: Contains perplexity scores for the validation dataset.

## Transformer Model

- **`q3.py`**: Main script for:
  - Training a new Transformer model.

  **Usage:**
  - **Train from Scratch**: `python3 q3.py`
    - This script will preprocess the data, train the Transformer model, and save perplexity scores for different datasets.

- **`Q3.ipynb`**: Jupyter notebook detailing the entire workflow for the Transformer model, including data preprocessing, model training, and evaluation.

- **`2021101075_LM3_test_perplexity.txt`**: Contains perplexity scores for the test dataset.
- **`2021101075_LM3_train_perplexity.txt`**: Contains perplexity scores for the training dataset.
- **`2021101075_LM3_val_perplexity.txt`**: Contains perplexity scores for the validation dataset.