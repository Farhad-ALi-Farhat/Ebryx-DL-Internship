# Forward and Backward Propagation

## What is Propagation?

Propagation refers to the process of passing information through a neural network during training.

There are two types:

1. **Forward Propagation** – Computes the model's prediction.
2. **Backward Propagation (Backpropagation)** – Updates the model's weights by propagating the error backward.

Together, these two processes allow a neural network to learn from data.

---

# Training Workflow

```text
Input Data
     │
     ▼
Forward Propagation
     │
     ▼
Prediction
     │
     ▼
Loss Function
     │
     ▼
Backward Propagation
     │
     ▼
Optimizer Updates Weights
     │
     ▼
Repeat
```

Each training iteration (or epoch) consists of these steps until the model learns the optimal weights.

---

# Forward Propagation

**Forward Propagation** is the process of moving data **from the input layer to the output layer** to generate a prediction.

During this process, each neuron:

1. Receives inputs.
2. Computes a weighted sum.
3. Adds a bias.
4. Applies an activation function.
5. Passes the output to the next layer.

---

# Step-by-Step Process

```text
Input
  │
  ▼
Multiply by Weights
  │
  ▼
Add Bias
  │
  ▼
Activation Function
  │
  ▼
Output
```

---

# Mathematical Representation

For a single neuron:

### Step 1: Compute the weighted sum

$$
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

Where:

- $x$ = Input
- $w$ = Weight
- $b$ = Bias

### Step 2: Apply the activation function

$$
a = f(z)
$$

Where:

- $f$ = Activation function
- $a$ = Output of the neuron

The output then becomes the input for the next layer.

---

# Example of Forward Propagation

Suppose:

- Input: $x = 4$
- Weight: $w = 2$
- Bias: $b = 1$

Step 1:

$$
z = (2 \times 4) + 1 = 9
$$

Step 2 (using ReLU):

$$
f(9) = 9
$$

The neuron outputs **9**, which is passed to the next layer.

---

# What Happens After Forward Propagation?

After the prediction is generated, the model compares it with the actual value using a **loss function**.

Example:

```
Actual Value     = 1
Predicted Value  = 0.70

Loss = Calculated using MSE or Cross-Entropy
```

If the loss is high, the model needs to adjust its weights.

This is where **Backward Propagation** begins.

---

# Backward Propagation (Backpropagation)

**Backward Propagation**, or **Backpropagation**, is the process of moving **backward through the network** to calculate how much each weight contributed to the error.

It computes the gradients of the loss with respect to every weight and bias, allowing the optimizer to update them.

---

# Why is Backpropagation Needed?

Suppose the prediction is incorrect.

The model must answer two questions:

- Which weights caused the error?
- By how much should each weight be changed?

Backpropagation calculates these answers using gradients.

---

# How Backpropagation Works

```text
Prediction
     │
     ▼
Calculate Loss
     │
     ▼
Compute Gradients
     │
     ▼
Propagate Error Backward
     │
     ▼
Update Weights
```

This process starts at the **output layer** and moves backward toward the **input layer**.

---

# Gradient

A **gradient** measures how much the loss changes when a weight changes.

- Large gradient → Large weight update.
- Small gradient → Small weight update.
- Zero gradient → No update.

The optimizer uses these gradients to minimize the loss.

---

# Weight Update

After computing the gradients, the optimizer updates the weights using Gradient Descent.

The update rule is:

$$
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
$$

Where:

- $w$ = Weight
- $\eta$ = Learning rate
- $L$ = Loss function
- $\frac{\partial L}{\partial w}$ = Gradient of the loss with respect to the weight

This process is repeated for every trainable parameter in the network.

---

# The Role of the Chain Rule

Backpropagation relies on the **Chain Rule** from calculus to compute gradients through multiple layers.

The Chain Rule allows the model to determine how each weight in earlier layers affects the final loss.

Although the mathematics can become complex for deep networks, modern frameworks such as TensorFlow and PyTorch compute these gradients automatically using **automatic differentiation (autograd)**.

---

# Forward vs Backward Propagation

| Feature | Forward Propagation | Backward Propagation |
|----------|---------------------|----------------------|
| Direction | Input → Output | Output → Input |
| Purpose | Generate predictions | Compute gradients and update weights |
| Uses Activation Functions | Yes | Yes (through their derivatives) |
| Uses Loss Function | No | Yes |
| Updates Weights | No | Yes (with an optimizer) |

---

# Complete Training Cycle

```text
Input Data
      │
      ▼
Forward Propagation
      │
      ▼
Prediction
      │
      ▼
Loss Function
      │
      ▼
Backward Propagation
      │
      ▼
Compute Gradients
      │
      ▼
Optimizer (SGD/Adam)
      │
      ▼
Update Weights
      │
      ▼
Repeat for Many Epochs
```

Over many training iterations, the loss decreases and the model's predictions become more accurate.

---

# Key Takeaways

- **Forward Propagation** moves information from the input layer to the output layer to generate predictions.
- Each neuron computes a weighted sum, adds a bias, and applies an activation function.
- The **loss function** measures the difference between the predicted and actual values.
- **Backward Propagation** computes gradients by propagating the error backward through the network.
- The **Chain Rule** enables gradients to be calculated across multiple layers.
- The optimizer uses these gradients to update the weights and biases, gradually minimizing the loss.
- Forward and backward propagation together form the core learning process of every neural network.
