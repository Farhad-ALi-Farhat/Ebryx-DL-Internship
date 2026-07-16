# Loss Functions (MSE & Cross-Entropy)

## What is a Loss Function?

A **Loss Function** measures **how far a model's predictions are from the actual values**.

It tells the neural network **how bad its predictions are** during training.

- **Lower loss** → Better predictions.
- **Higher loss** → Poor predictions.

The objective of training is to **minimize the loss** by updating the model's weights and biases.

---

# Why Do We Need a Loss Function?

During training, the neural network:

1. Makes predictions.
2. Compares predictions with the actual values.
3. Calculates the loss.
4. Updates the weights using backpropagation and an optimizer.
5. Repeats the process until the loss is minimized.

Without a loss function, the model would have no way to measure its performance or improve.

---

# Common Loss Functions

The most commonly used loss functions are:

- **Mean Squared Error (MSE)** → Regression
- **Cross-Entropy Loss** → Classification

---

# Mean Squared Error (MSE)

Mean Squared Error is primarily used for **regression problems**, where the target value is continuous (e.g., predicting house prices, temperatures, or salaries).

It calculates the average of the squared differences between the actual and predicted values.

## Formula
$$
\[
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]
$$

Where:

$$
- \(y_i\) = Actual value
- \(\hat{y}_i\) = Predicted value
- \(n\) = Number of samples
$$

---

# Example of MSE

Suppose the actual and predicted values are:

| Actual | Predicted |
|---------|-----------|
| 10 | 8 |
| 20 | 18 |
| 30 | 35 |

### Step 1: Calculate Errors

| Actual | Predicted | Error |
|---------|-----------|-------|
| 10 | 8 | 2 |
| 20 | 18 | 2 |
| 30 | 35 | -5 |

### Step 2: Square Errors

| Error | Squared Error |
|-------|---------------|
| 2 | 4 |
| 2 | 4 |
| -5 | 25 |

### Step 3: Average

$$
\[
MSE=\frac{4+4+25}{3}=11
\]
$$

The **MSE = 11**.

---

# Why Square the Errors?

Squaring the errors has two benefits:

- Makes all errors positive.
- Penalizes larger errors more heavily than smaller ones.

For example:

```
Error = 2  → 4
Error = 5  → 25
```

Large prediction errors have a much greater impact on the loss.

---

# Advantages of MSE

- Easy to compute.
- Differentiable (required for gradient descent).
- Penalizes large errors.
- Widely used in regression tasks.

---

# Disadvantages of MSE

- Sensitive to outliers.
- Loss values can become very large if predictions are far from the actual values.

---

# Cross-Entropy Loss

Cross-Entropy Loss is commonly used for **classification problems**.

Instead of measuring numerical differences, it measures **how well the predicted probability distribution matches the true labels**.

The goal is to assign a **high probability to the correct class** and a **low probability to the incorrect classes**.

---

# Binary Cross-Entropy

Used for **binary classification**, where there are only two classes (e.g., spam vs. not spam, real vs. fake).

## Formula

$$
\[
L = -\left(y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right)
\]
$$

Where:

$$
- \(y\) = Actual label (0 or 1)
- \(\hat{y}\) = Predicted probability
$$

---

# Example (Binary Classification)

Suppose:

```
Actual Label = 1
Predicted Probability = 0.9
```

The prediction is close to the correct label, so the loss is **small**.

Now suppose:

```
Actual Label = 1
Predicted Probability = 0.1
```

The prediction is far from the correct label, so the loss is **large**.

---

# Categorical Cross-Entropy

Used for **multi-class classification**, where there are more than two classes (e.g., cat, dog, bird).

The formula is:

$$
\[
L=-\sum y_i\log(\hat{y}_i)
\]
$$

This loss function is typically used with the **Softmax activation function**, which converts model outputs into probabilities that sum to 1.

---

# Why Use Cross-Entropy?

Cross-Entropy encourages the model to assign **higher probabilities to the correct class**.

Example:

```
Correct Class = Cat

Prediction 1:
Cat = 0.95
Dog = 0.03
Bird = 0.02

→ Low Loss
```

```
Prediction 2:
Cat = 0.20
Dog = 0.50
Bird = 0.30

→ High Loss
```

The closer the predicted probability is to the true class, the lower the loss.

---

# MSE vs Cross-Entropy

| Feature | MSE | Cross-Entropy |
|----------|-----|---------------|
| Used For | Regression | Classification |
| Output Type | Continuous values | Probabilities |
| Goal | Minimize prediction error | Maximize probability of the correct class |
| Sensitive to Large Errors | Yes | Focuses on prediction confidence |
| Common Output Activation | Linear | Sigmoid (Binary) / Softmax (Multi-Class) |

---

# Choosing the Right Loss Function

| Problem Type | Recommended Loss Function |
|--------------|---------------------------|
| House Price Prediction | Mean Squared Error (MSE) |
| Temperature Prediction | Mean Squared Error (MSE) |
| Binary Classification | Binary Cross-Entropy |
| Multi-Class Classification | Categorical Cross-Entropy |

---

# Key Takeaways

- A **loss function** measures how well a neural network performs during training.
- The objective of training is to **minimize the loss**.
- **Mean Squared Error (MSE)** is mainly used for **regression** tasks and penalizes larger errors more heavily.
- **Binary Cross-Entropy** is used for **binary classification** tasks.
- **Categorical Cross-Entropy** is used for **multi-class classification** tasks, usually with the **Softmax** activation function.
- Choosing the appropriate loss function depends on the type of machine learning problem being solved.
