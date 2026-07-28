# Optimizers (SGD & Adam)

## What is an Optimizer?

An **optimizer** is an algorithm that updates the **weights** and **biases** of a neural network to minimize the **loss function**.

After each forward pass:

1. The model makes predictions.
2. The loss is calculated.
3. Backpropagation computes the gradients.
4. The optimizer updates the weights to reduce the loss.

The goal is to find the set of weights that produces the **lowest possible loss**.

---

# Training Process

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
Backpropagation
     │
     ▼
Optimizer Updates Weights
     │
     ▼
Repeat
```

---

# Gradient Descent

Most optimizers are based on **Gradient Descent**.

Gradient Descent minimizes the loss by moving the weights in the direction where the loss decreases the fastest.

The weight update rule is:

$$
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
$$

Where:

- $w$ = Weight
- $\eta$ = Learning rate
- $L$ = Loss function
- $\frac{\partial L}{\partial w}$ = Gradient of the loss with respect to the weight

---

# Learning Rate

The **learning rate** determines the size of each weight update.

- Small learning rate → Slow learning
- Large learning rate → May overshoot the minimum
- Appropriate learning rate → Faster and stable convergence

Example:

```text
Learning Rate = 0.001
↓
Small, stable updates

Learning Rate = 1.0
↓
Large updates (may miss the optimum)
```

---

# Stochastic Gradient Descent (SGD)

**Stochastic Gradient Descent (SGD)** updates the model parameters using **one training example (or a small mini-batch)** at a time.

Unlike Batch Gradient Descent, which uses the entire dataset before updating weights, SGD performs frequent updates.

Weight update:

$$
w = w - \eta \nabla L
$$

Where:

- $\eta$ = Learning rate
- $\nabla L$ = Gradient of the loss

---

# How SGD Works

```text
Training Sample
      │
      ▼
Forward Pass
      │
      ▼
Compute Loss
      │
      ▼
Compute Gradient
      │
      ▼
Update Weights
      │
      ▼
Next Sample
```

---

# Advantages of SGD

- Simple to implement.
- Uses less memory.
- Faster updates.
- Works well with large datasets.
- Can escape shallow local minima due to noisy updates.

---

# Disadvantages of SGD

- Training can be noisy.
- May converge slowly.
- Sensitive to the learning rate.
- May oscillate around the minimum instead of settling quickly.

---

# Adam Optimizer

**Adam (Adaptive Moment Estimation)** is one of the most popular optimizers used in deep learning.

It combines ideas from:

- **Momentum**
- **RMSProp**

Adam adapts the learning rate for each parameter individually, leading to faster and more stable convergence.

Instead of using a fixed learning rate for every weight, Adam automatically adjusts it during training.

---

# How Adam Works

Adam keeps track of:

- **First Moment (Mean of Gradients)**

$$
m_t
$$

- **Second Moment (Variance of Gradients)**

$$
v_t
$$

The parameter update is:

$$
w = w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

Where:

- $\hat{m}_t$ = Bias-corrected first moment estimate
- $\hat{v}_t$ = Bias-corrected second moment estimate
- $\epsilon$ = Small constant to prevent division by zero

You don't need to memorize this formula, but it's useful to understand that Adam uses both the **average gradient** and the **variance of gradients** to make smarter updates.

---

# Advantages of Adam

- Fast convergence.
- Automatically adjusts learning rates.
- Less sensitive to the initial learning rate.
- Works well for deep neural networks.
- Performs well on sparse and noisy data.
- One of the most commonly used optimizers in practice.

---

# Disadvantages of Adam

- Requires more memory than SGD.
- More computationally expensive.
- May not generalize as well as SGD in some cases.

---

# SGD vs Adam

| Feature | SGD | Adam |
|----------|-----|------|
| Learning Rate | Fixed | Adaptive |
| Speed | Slower | Faster |
| Memory Usage | Low | Higher |
| Convergence | May require tuning | Usually converges quickly |
| Ease of Use | More tuning required | Works well with default settings |
| Common Usage | Traditional ML & some DL tasks | Most modern deep learning models |

---

# Which Optimizer Should You Use?

| Situation | Recommended Optimizer |
|-----------|-----------------------|
| Learning deep learning fundamentals | SGD |
| Image classification | Adam |
| Natural Language Processing (NLP) | Adam |
| Computer Vision | Adam |
| Large neural networks | Adam |
| Research on model generalization | SGD |

---

# Key Takeaways

- An **optimizer** updates the weights and biases of a neural network to minimize the loss function.
- **Gradient Descent** is the foundation of many optimization algorithms.
- The **learning rate** controls how large each update is during training.
- **Stochastic Gradient Descent (SGD)** is simple, memory-efficient, and performs frequent weight updates.
- **Adam (Adaptive Moment Estimation)** combines Momentum and RMSProp to provide adaptive learning rates for each parameter.
- **Adam** is the default choice for many modern deep learning applications due to its fast convergence and ease of use, while **SGD** remains valuable for its simplicity and, in some cases, better generalization.
