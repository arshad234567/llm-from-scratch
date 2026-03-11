# Mini GPT (Character-Level Language Model) – PyTorch

## Overview

This project implements a **Mini GPT (Generative Pretrained Transformer)** from scratch using **PyTorch**.
The model is a **decoder-only Transformer architecture** trained on the **Shakespeare dataset** to generate text character by character.

The purpose of this project is to understand the **core architecture of modern Large Language Models (LLMs)** such as GPT-style models.

The model learns patterns from text and generates new text by predicting the **next character given previous characters**.

---

# Project Workflow

The overall pipeline of the model is:

```
Raw Text Dataset
        ↓
Character Tokenization
        ↓
Embedding Layer
        ↓
Positional Encoding
        ↓
Transformer Blocks
        ↓
Language Model Head
        ↓
Next Character Prediction
        ↓
Text Generation
```

---

# Dataset

The model is trained on a **Shakespeare text dataset**.

Example training text:

```
To be or not to be that is the question
```

The dataset is converted into **character tokens** so the model can process it numerically.

---

# Tokenization

Each character in the dataset is mapped to an integer.

Example:

```
Text: "hello"

Tokens:
h → 7
e → 3
l → 10
l → 10
o → 14
```

Two dictionaries are created:

```
stoi → character to index
itos → index to character
```

This allows the model to convert between text and tokens.

---

# Train / Validation Split

The dataset is split into:

```
90% training data
10% validation data
```

Purpose:

* Train the model on most of the dataset
* Evaluate performance on unseen data

---

# Training Samples

The model learns **next-token prediction**.

Example:

Input sequence:

```
hello wor
```

Target sequence:

```
ello worl
```

The model learns to predict the **next character at every position**.

---

# Transformer Architecture

The model uses a **decoder-only Transformer architecture**, similar to GPT models.

Each Transformer block contains:

```
LayerNorm
↓
Multi-Head Self Attention
↓
Residual Connection
↓
LayerNorm
↓
Feed Forward Network
↓
Residual Connection
```

Multiple blocks are stacked to build deep representations.

---

# Self-Attention Mechanism

Self-attention allows each token to **focus on relevant tokens in the sequence**.

Example sentence:

```
"The cat sat on the mat"
```

The word **sat** may attend to **cat**.

Steps of attention:

1. Compute Query, Key, Value vectors
2. Calculate attention scores
3. Apply scaling
4. Apply masking
5. Convert scores to probabilities using Softmax
6. Compute weighted sum of values

This produces a contextual representation of tokens.

---

# Masked Attention

Because this is a **language generation model**, it cannot see future tokens.

Example:

```
Input: hello
Predict: next character
```

The model should **not look ahead in the sequence**.

This is achieved using a **causal mask**.

---

# Multi-Head Attention

Instead of one attention mechanism, the model uses **multiple attention heads**.

Each head can learn different relationships:

Example:

| Head   | Learns                     |
| ------ | -------------------------- |
| Head 1 | syntax                     |
| Head 2 | semantic relationships     |
| Head 3 | long-distance dependencies |

The outputs of all heads are concatenated.

---

# Feed Forward Network

After attention, each token passes through a **fully connected neural network**:

```
Linear Layer
↓
ReLU Activation
↓
Linear Layer
```

Purpose:

* Introduce non-linearity
* Transform representations

---

# Embeddings

Two types of embeddings are used:

### Token Embeddings

Convert token IDs into vectors.

Example:

```
token → embedding vector
```

### Positional Embeddings

Transformers do not inherently understand word order.

Positional embeddings add position information:

```
embedding + positional encoding
```

This helps the model understand sequence order.

---

# Language Model Head

The final layer converts embeddings into **vocabulary probabilities**.

Example:

```
Input: "the cat"

Prediction probabilities:

sat → 0.60
ran → 0.20
slept → 0.20
```

The token with highest probability is selected.

---

# Loss Function

The model uses **Cross Entropy Loss**.

Purpose:

Measure how different the predicted distribution is from the true token.

Training objective:

```
Minimize prediction error
```

---

# Training Loop

The training process follows:

```
1. Sample batch of sequences
2. Forward pass through the model
3. Compute loss
4. Backpropagation
5. Update model parameters
```

This process repeats for thousands of iterations.

---

# Text Generation

After training, the model can generate new text.

Generation process:

```
Start with a prompt
↓
Predict next character
↓
Append prediction to sequence
↓
Repeat
```

Example:

```
Prompt: "The king"

Generated text:
"The king was a noble man who ruled the kingdom..."
```

---

# Model Components

The model consists of:

```
Token Embedding Layer
Positional Embedding Layer
Transformer Blocks
Layer Normalization
Language Model Head
```

---

# Hyperparameters

Example configuration used:

```
batch_size = 32
block_size = 64
embedding_dimension = 128
number_of_heads = 4
number_of_layers = 4
dropout = 0.2
learning_rate = 3e-4
```

---

# Repository Structure

```
mini-gpt-pytorch
│
├── notebooks
│   └── mini_gpt.ipynb
│
├── models
│   └── mini_gpt_model.pth
│
├── data
│   └── shakespeare.txt
│
├── src
│   └── transformer_components.py
│
└── README.md
```

---

# Applications

This architecture forms the foundation of modern language models such as:

* GPT-style text generation models
* conversational AI systems
* text summarization systems
* machine translation models
* code generation systems

---

# Learning Outcomes

This project demonstrates understanding of:

* Transformer architecture
* Self-attention mechanism
* Multi-head attention
* Token embeddings
* Positional encoding
* Language model training
* Text generation

---

# Future Improvements

Possible extensions:

* train on larger datasets
* implement Byte Pair Encoding tokenizer
* add evaluation metrics
* deploy as a web application
* scale to larger models

---

# License

This project is intended for educational purposes to understand the fundamentals of Transformer-based language models.
