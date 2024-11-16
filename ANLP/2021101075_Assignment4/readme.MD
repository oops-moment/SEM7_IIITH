# Model Quantization and Compression Assignment

This repository contains the implementation of various quantization techniques applied to language models, comparing different approaches and their impact on model performance.

## Project Structure

```
.
├── part1.py           # Implementation of custom quantization
├── part2.py           # Implementation using bitsandbytes
├── README.md          # This file
└── report.pdf         # Detailed analysis and findings
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- NumPy
- Matplotlib
- tqdm
- psutil
- bitsandbytes

Install dependencies:
```bash
pip install torch transformers datasets numpy matplotlib tqdm psutil bitsandbytes
```

## Running the Code

### Part 1: Custom Quantization

```bash
python part1.py
```

This script implements:
- Whole-model 8-bit quantization
- Selective component quantization
- Performance analysis and visualization

### Part 2: Bitsandbytes Integration

```bash
python part2.py
```

This script implements:
- 8-bit quantization using bitsandbytes
- 4-bit linear quantization
- NF4 quantization
- Comparative analysis

## Dataset

The code uses the Wikipedia dataset (20220301.en) with 3000 samples for evaluation. The dataset is automatically downloaded through the Hugging Face datasets library.

## Model

The implementation uses the GPT-2 model from Hugging Face's model hub. The same model is used consistently across all experiments for fair comparison.

## Output

Both scripts generate:
- Memory usage metrics
- Inference latency measurements
- Perplexity scores
- Visualization plots for comparison

Results are displayed in the console and saved as plots.

## Notes

- Ensure sufficient GPU memory is available
- The first run may take longer due to dataset and model downloads
- For optimal performance, run on a CUDA-enabled GPU
- You can obtain models from this link [Models](https://drive.google.com/drive/folders/1ELHjeVg9AtipnU5pD7TQ6CFFTE9bRiaZ?usp=share_link)