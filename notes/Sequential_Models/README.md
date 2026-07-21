# Sequential Models (RNNs, LSTMs, GRUs)

This directory contains notes on **Sequential Models**, a family of neural network architectures designed to process **ordered data** such as text, speech, time series, and video. Unlike feedforward neural networks, sequential models can capture relationships between elements in a sequence by maintaining contextual information across time steps.

These notes introduce the fundamentals of sequential data, explain how **Recurrent Neural Networks (RNNs)** work, discuss their limitations, and explore improved architectures such as **Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRUs)**. The directory also covers common sequence modeling patterns used in real-world deep learning applications.

---

## Topics Covered

### 1. Sequential Data Overview
- What is sequential data?
- Characteristics of sequential data
- Types of sequential data
- Sequential data vs. tabular data
- Applications of sequential data

### 2. Recurrent Neural Networks (RNNs)
- Introduction to RNNs
- Hidden state and memory
- Unrolled RNN architecture
- Forward propagation through time
- Advantages of RNNs
- Limitations of vanilla RNNs
  - Vanishing gradients
  - Exploding gradients
  - Short-term memory
  - Sequential computation

### 3. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)
- Why LSTMs and GRUs were introduced
- LSTM architecture
  - Cell state
  - Forget gate
  - Input gate
  - Output gate
- GRU architecture
  - Update gate
  - Reset gate
- Comparison of LSTM and GRU
- TensorFlow/Keras implementation examples

### 4. Sequence Modeling Patterns
- One-to-One
- One-to-Many
- Many-to-One
- Many-to-Many (Synchronous)
- Many-to-Many (Encoder-Decoder)
- Real-world applications of each pattern

---

## Learning Outcomes

After completing these notes, you should be able to:

- Understand the characteristics of sequential data.
- Explain why traditional feedforward neural networks are not suitable for sequence modeling.
- Describe how Recurrent Neural Networks (RNNs) process sequential information.
- Explain the vanishing and exploding gradient problems.
- Understand how LSTMs and GRUs overcome the limitations of vanilla RNNs.
- Compare RNNs, LSTMs, and GRUs based on architecture, performance, and use cases.
- Identify different sequence modeling patterns and their applications.
- Build basic RNN, LSTM, and GRU models using TensorFlow/Keras.

---

## Prerequisites

Before studying these notes, you should be familiar with:

- Artificial Neural Networks (ANNs)
- Forward and Backpropagation
- Gradient Descent
- Activation Functions
- Loss Functions
- TensorFlow/Keras fundamentals

---

## Technologies

- Python
- TensorFlow
- Keras
- NumPy

---

## What's Next?

The next topic in the Deep Learning roadmap is **Attention Mechanisms and Transformers**, where you'll learn how modern architectures overcome the limitations of recurrent networks using self-attention. Topics include:

- Attention Mechanism
- Self-Attention
- Positional Encoding
- Transformer Architecture
- BERT
- GPT
- Vision Transformers (ViTs)

These concepts form the foundation of modern Natural Language Processing (NLP), Large Language Models (LLMs), and many state-of-the-art AI systems.
