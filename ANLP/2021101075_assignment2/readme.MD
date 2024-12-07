# Transformer Model Implementation

## Installation

To install the required modules, run:

pip install -r requirements.txt

-In order to test the model, the model needs to be downloaded from [this link](https://drive.google.com/drive/folders/1uRdlSqOGf0mOK81_-JXHuFAXXquAj5Cq?usp=sharing).

## Files in the Project

- **ANLP_2021101075.ipynb**: Complete Jupyter notebook that includes all executed cells, detailing the model training and evaluation.

- **ANLP_2021101075.py**: Python source file that trains the model from scratch, runs testing on the test loader, and saves the trained model. Running this file will initiate the training process and save the final trained model.

- **decoder.py**: Implementation of the decoder module for the transformer architecture.

- **encoder.py**: Implementation of the encoder module for the transformer, including components such as the feed-forward network, residual connections, and other necessary functions required by the encoder.

- **test.py**: Loads the pretrained model and performs inference on the test dataset.

- **train.py**: Script to train the model from scratch, handling multiple iterations of training.

- **utils.py**: Contains utility functions used by both the encoder and decoder, along with functions to prepare datasets, create data loaders, and preprocess the data.

- **testbleu.txt**: Contains the BLEU and ROUGE scores for all 1305 sentences from the test dataset, along with the final averaged BLEU score.

- **trainbleu.txt**: Stores the BLEU and ROUGE scores for the sentences in the training dataset, including the final average scores.

- **transformer.pt**: The trained model file which can be loaded as per the instructions in test.py.

- **report.pdf**: A complete document including hyperparameter tuning results and answers to theoretical questions.

## Note

You need to move the files `test.en` and `test.fr` to the project directory in order to test the code. For training, place the files `train.en`, `train.fr`, `dev.en`, and `dev.fr` in the directory. The model will be trained and saved as `transformer.pt`.

- In order to train, run `python3 train.py`.
- In order to test, run `python test.py`.
