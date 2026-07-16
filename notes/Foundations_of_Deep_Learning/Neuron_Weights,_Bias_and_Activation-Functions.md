# Neuron, Weights, Bias, and Activation Functions

## Artificial Neuron

An **artificial neuron** is the basic building block of a neural network. It mimics the behavior of a biological neuron by receiving inputs, processing them, and producing an output.

Each neuron performs three main operations:

1. Receives input values.
2. Calculates a weighted sum of the inputs.
3. Applies an activation function to produce the output.

---

# Structure of a Neuron

```text
           x₁ ──(w₁)──┐
                      │
           x₂ ──(w₂)──┤
                      │
           x₃ ──(w₃)──┤──► Weighted Sum + Bias ─► Activation Function ─► Output
                      │
           xn ──(wn)──┘
```

Where:

- **x** = Input features
- **w** = Weights
- **b** = Bias
- **f()** = Activation function
- **y** = Output

---

# Mathematical Representation

A neuron first computes the weighted sum:

\[
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
\]

Then applies an activation function:

\[
a = f(z)
\]

Where:

- **z** = Weighted sum
- **a** = Final output of the neuron

---

# Weights

Weights determine **how important each input is**.

- Larger weight → Greater influence on the output.
- Smaller weight → Less influence.
- Negative weight → Decreases the output.

Example:

| Input | Weight | Contribution |
|--------|--------|-------------|
| 5 | 0.8 | 4.0 |
| 2 | 0.1 | 0.2 |
| 7 | -0.5 | -3.5 |

The neural network learns the optimal weights during training.

---

# Bias

A **bias** is an additional parameter added to the weighted sum.

It allows the neuron to shift the activation function left or right, making the model more flexible.

Without bias:

\[
z = \sum wx
\]

With bias:

\[
z = \sum wx + b
\]

Think of bias as the **intercept** in a linear equation.

Example:

```
Without bias:
y = 2x

With bias:
y = 2x + 3
```

Bias helps the neuron make predictions even when all input values are zero.

---

# Activation Function

The activation function determines whether a neuron should be activated.

Without an activation function, a neural network would simply perform linear transformations and could not learn complex nonlinear patterns.

Activation functions introduce **non-linearity**, allowing neural networks to solve complex problems.

---

# Why Do We Need Activation Functions?

Suppose we stack many layers without activation functions.

Each layer performs:

```
Output = Weight × Input + Bias
```

Stacking multiple linear layers still results in another linear function.

Therefore, no matter how many layers we add, the network behaves like a simple linear model.

Activation functions solve this problem by introducing **non-linearity**, enabling deep neural networks to learn complex relationships.

---

# ReLU (Rectified Linear Unit)

The most commonly used activation function in hidden layers.

## Formula

\[
f(x)=\max(0,x)
\]

### Output

```
Input < 0  → 0

Input ≥ 0 → x
```

### Graph

```text
Output
 ^
 |        /
 |      /
 |    /
 |  /
 |/
 +-----------------> Input
```

### Advantages

- Simple and computationally efficient.
- Reduces the vanishing gradient problem.
- Trains deep networks faster.
- Most popular choice for hidden layers.

### Disadvantages

- Neurons can "die" if they always receive negative inputs (Dead ReLU problem).

---

# Sigmoid

The Sigmoid function squashes values between **0 and 1**.

## Formula

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

### Output Range

```
0 to 1
```

### Graph

```text
1.0 |           ______
    |         /
0.5 |-------/
    |     /
0.0 |____/
      Input
```

### Advantages

- Produces probabilities.
- Smooth and differentiable.
- Useful for binary classification output layers.

### Disadvantages

- Suffers from the vanishing gradient problem.
- Slow convergence in deep networks.
- Outputs are not zero-centered.

---

# Tanh (Hyperbolic Tangent)

Tanh is similar to Sigmoid but outputs values between **-1 and 1**.

## Formula

\[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\]

### Output Range

```
-1 to 1
```

### Graph

```text
 1 |        ______
   |      /
 0 |-----/
   |    /
-1 |___/
      Input
```

### Advantages

- Zero-centered output.
- Stronger gradients than Sigmoid.
- Often performs better than Sigmoid in hidden layers.

### Disadvantages

- Still suffers from the vanishing gradient problem.
- Less commonly used than ReLU in modern deep networks.

---

# Comparison of Activation Functions

| Feature | ReLU | Sigmoid | Tanh |
|----------|------|----------|------|
| Output Range | 0 to ∞ | 0 to 1 | -1 to 1 |
| Zero-Centered | No | No | Yes |
| Vanishing Gradient | Rare | Yes | Yes |
| Training Speed | Fast | Slow | Moderate |
| Common Usage | Hidden Layers | Binary Output Layer | Sometimes Hidden Layers |

---

# Which Activation Function Should You Use?

| Layer | Recommended Activation |
|--------|------------------------|
| Hidden Layers | ReLU |
| Binary Classification Output | Sigmoid |
| Multi-Class Classification Output | Softmax (covered later) |
| Regression Output | Linear (No activation) |

---

# Key Takeaways

- A **neuron** is the fundamental unit of a neural network.
- **Weights** determine the importance of each input.
- **Bias** shifts the neuron's output and improves learning flexibility.
- **Activation functions** introduce non-linearity, allowing neural networks to learn complex patterns.
- **ReLU** is the most widely used activation function for hidden layers due to its simplicity and efficiency.
- **Sigmoid** is commonly used in the output layer for binary classification tasks.
- **Tanh** outputs values between -1 and 1 and generally performs better than Sigmoid in hidden layers, though it is less common than ReLU.
