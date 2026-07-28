# Introduction to Deep Learning and Neural Networks

## What is Deep Learning?

**Deep Learning** is a subset of **Machine Learning (ML)** that uses **Artificial Neural Networks (ANNs)** with multiple hidden layers to learn complex patterns from data.

It is inspired by the human brain, where neurons work together to process and transmit information.

Unlike traditional machine learning, deep learning automatically learns useful features from raw data instead of relying on manually engineered features.

---

# What is a Neural Network?

A **Neural Network** is a computational model composed of interconnected units called **neurons**.

These neurons are organized into three types of layers:

- **Input Layer** – Receives the input data.
- **Hidden Layer(s)** – Process the input and learn patterns.
- **Output Layer** – Produces the final prediction.

### Basic Structure

```text
Input Layer        Hidden Layer        Output Layer

 x1 ───┐
        ├────●────┐
 x2 ───┘          │
                  ├────●──► Prediction
 x3 ───┐          │
        ├────●────┘
 x4 ───┘
```

Each connection has an associated **weight**, which determines the importance of that connection.

---

# How Does a Neuron Work?

A neuron performs three basic operations:

1. Receives input values.
2. Computes a weighted sum of the inputs.
3. Applies an activation function to produce an output.

Mathematically:

\[
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
\]

Where:

- \(x\) = input
- \(w\) = weights
- \(b\) = bias

The output is then computed as:

\[
a = f(z)
\]

Where:

- \(f()\) = activation function
- \(a\) = neuron output

---

# Why is it Called "Deep" Learning?

The term **deep** refers to the number of hidden layers.

- **No hidden layer** → Perceptron
- **One hidden layer** → Shallow Neural Network
- **Multiple hidden layers** → Deep Neural Network (DNN)

More hidden layers allow the network to learn increasingly complex features.

Example:

```text
Image
  ↓
Edges
  ↓
Shapes
  ↓
Objects
  ↓
Prediction
```

---

# Traditional Machine Learning vs Deep Learning

| Traditional Machine Learning | Deep Learning |
|------------------------------|---------------|
| Manual feature engineering | Learns features automatically |
| Works well with structured data | Excellent for unstructured data |
| Performs well on smaller datasets | Usually requires large datasets |
| Faster training | Longer training time |
| Easier to interpret | Less interpretable ("black box") |

---

# Applications of Deep Learning

Deep learning is widely used in:

- Image classification
- Object detection
- Face recognition
- Speech recognition
- Natural Language Processing (NLP)
- Machine translation
- Chatbots and virtual assistants
- Recommendation systems
- Autonomous vehicles
- Medical image analysis
- Fraud detection

---

# Advantages

- Automatically learns useful features.
- Handles highly complex and nonlinear problems.
- Excellent performance on images, text, audio, and video.
- Often achieves state-of-the-art accuracy.

---

# Limitations

- Requires large amounts of training data.
- Computationally expensive.
- Needs GPUs/TPUs for efficient training.
- Longer training times.
- Harder to interpret than traditional ML models.
- Can overfit if not properly regularized.

---

# Popular Deep Learning Frameworks

Some commonly used frameworks are:

- TensorFlow
- Keras
- PyTorch
- JAX

---

# Key Takeaways

- Deep Learning is a subset of Machine Learning.
- It uses Artificial Neural Networks with multiple hidden layers.
- Neural networks automatically learn features from data.
- A neuron computes a weighted sum of inputs, adds a bias, and applies an activation function.
- Deep learning excels at solving problems involving images, text, audio, and video.
- Although powerful, deep learning requires significant data and computational resources.

---

# What's Next?

The next topics are typically:

1. Perceptron
2. Activation Functions
3. Forward Propagation
4. Loss Functions
5. Backpropagation
6. Gradient Descent
7. Optimizers (SGD, Adam, RMSprop)
8. Multi-Layer Perceptrons (MLPs)
9. Training Neural Networks
10. Regularization Techniques
