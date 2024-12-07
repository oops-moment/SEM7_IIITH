# Transformer Model Implementation

## Installation

To install the required modules, run:

pip install -r requirements.txt

- In order to test the model, the model needs to be downloaded from [this link](https://drive.google.com/drive/folders/1HjKoGE0amrxcLDZkhi9yc244pAsEK5aU?usp=sharing).

Note that the model named ‘Summarize_weights1.pth’ is for Question 1, while the models named ‘checkpoint-2400’ and ‘checkpoint-5300’ are for Questions 3 and 2, respectively. Accordingly, they should be downloaded as needed. Additionally, please note that these models are saved from the Trainer, so they include configuration files and need to be loaded accordingly

## Files in the Project

- **Question1/Question1.ipynb**: A complete executable notebook that includes the run for fine-tuning the GPT-2 small model with soft prompt tuning, using the [summarize] token.

- **Question2/Question2.ipynb**: A complete executable notebook that runs fine-tuning of the GPT-2 small model using the LoRA configuration. Note that the hyperparameter rank was set to 4 in this case, and the run follows the specified LoRA configuration.

- **Question3/Question3.ipynb**: A complete executable notebook for fine-tuning only the last layer (in this case, the LM head) of the GPT-2 model, as specified for Question 3.


- **Question1/Question1.py**: Corresponding to the .ipynb notebook, this is the converted Python script for this question. Note that it is preferable to run the notebook (.ipynb) version, as this script is simply a converted version of that.

- **Question2/Question2.py**: Corresponding to the .ipynb notebook, this is the converted Python script for this question. Note that it is preferable to run the notebook (.ipynb) version, as this script is simply a converted version of that.

- **Question3/Question3.py**: Corresponding to the .ipynb notebook, this is the converted Python script for this question. Note that it is preferable to run the notebook (.ipynb) version, as this script is simply a converted version of that.
