# Next Token Prediction using Transformer Decoder

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains our implementation of a Transformer-based model focused on **next token prediction** using the **Masked Multi-Head Attention mechanism**. Developed as part of the **CS584 Machine Learning** course at Illinois Tech, this project demonstrates how decoder-only transformer architectures can be used for character-level language modeling using the `tiny_shakespeare` dataset.

---

## 📌 Project Overview

🔹 **Objective:**  
Implement and analyze the Transformer Decoder for next token prediction in an auto-regressive setting, focusing on character-level sequence modeling.

🔹 **Dataset:**  
[tiny_shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

🔹 **Key Feature:**  
Custom-built decoder-only transformer architecture utilizing masked attention to prevent information leakage from future tokens.

---

## 🧠 Model Architecture

- **Decoder Layers:** 6  
- **Attention Heads:** 6  
- **Embedding Size:** 384 → (Later tuned to 512)  
- **Attention Type:** Masked Multi-Head Self Attention  
- **Trainable Parameters:** ~19 million  
- **Loss Metric:** Cross-Entropy  
- **Evaluation Metric:** Perplexity  

---

## 🧪 Training Details

| Parameter        | Value       |
|------------------|-------------|
| Batch Size       | 64          |
| Block Size       | 256         |
| Max Iterations   | 2,500       |
| Learning Rate    | 0.0001      |
| Optimizer        | AdamW       |
| Eval Frequency   | Every 500 steps |

---

## 📉 Results

| Metric                | Initial  | Final  |
|------------------------|----------|--------|
| **Training Perplexity** | 62.26    | 27.42  |
| **Validation Perplexity** | 63.43    | 28.36  |

Despite strong improvements in perplexity, large-scale configurations encountered **GPU memory limitations** on both local (RTX 4060) and cloud (Google Colab) environments.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/transformer-decoder-next-token-prediction.git
   cd transformer-decoder-next-token-prediction
Install dependencies:
pip install -r requirements.txt

Launch Jupyter Notebook:
jupyter notebook transformer_model_v1_perplexity.ipynb


## Highlights:
Implemented a transformer decoder from scratch using PyTorch.

Integrated masked attention for causal language modeling.

Trained and evaluated using the tiny_shakespeare dataset.

Achieved perplexity under 28 despite resource constraints.

## Limitations
Training larger models exceeded GPU memory limits (on both RTX 4060 and Google Colab).

Needs further tuning or distributed training setup for improved scalability.

## Built With
Python 3.10
PyTorch
NumPy
Jupyter Notebook
Google Colab

## References
Attention is All You Need (Vaswani et al., NeurIPS 2017)
Tiny Shakespeare Dataset
3Blue1Brown: Neural Networks
