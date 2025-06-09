Next Token Prediction using Transformer Decoder (CS584 Machine Learning Project)
This repository contains the implementation and training of a Transformer-based model focused on next token prediction using the Masked Multi-Head Attention mechanism. Developed as part of the CS584 Machine Learning course at Illinois Tech, this project explores core concepts of the Transformer decoder architecture, focusing on sequence modeling for character-level language modeling.

📌 Project Overview
Objective: Implement and analyze the Transformer Decoder for next token prediction in an auto-regressive setting.

Dataset: tiny_shakespeare

Key Feature: Custom-built decoder-only transformer architecture utilizing masked attention to prevent information leakage from future tokens.

🧠 Model Architecture
Decoder Layers: 6

Attention Heads: 6

Embedding Size: 384 (later increased to 512)

Attention Type: Masked Multi-Head Self Attention

Total Parameters: Up to ~19 million

Training Loss Metric: Cross-Entropy Loss

Evaluation Metric: Perplexity

🧪 Training Details
Batch Size: 64

Block Size (Context Window): 256

Max Iterations: 2,500

Learning Rate: 0.0001

Optimizer: AdamW

Evaluation Frequency: Every 500 steps

📉 Results
Initial Training Perplexity: 62.26 → Final: 27.42

Initial Validation Perplexity: 63.43 → Final: 28.36

While performance improved, computational challenges (GPU memory constraints) were observed during complex model runs.

🔧 Limitations
Experiments on a laptop with RTX 4060 and Google Colab faced out-of-memory errors with the largest model configuration.

Further optimization and distributed training setups could improve scalability and stability.

📚 References
Attention Is All You Need – Vaswani et al. (NeurIPS 2017)

Tiny Shakespeare Dataset

3Blue1Brown - Neural Networks Visualizations
