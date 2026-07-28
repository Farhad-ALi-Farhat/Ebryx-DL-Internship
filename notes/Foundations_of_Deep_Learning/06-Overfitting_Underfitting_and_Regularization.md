# Overfitting, Underfitting, and Regularization

## Model Fitting

When training a machine learning or deep learning model, the goal is to learn patterns from the training data that generalize well to unseen data.

A model can fall into one of three categories:

- **Underfitting** – The model learns too little.
- **Good Fit** – The model learns the right patterns.
- **Overfitting** – The model learns too much, including noise.

---

# Underfitting

**Underfitting** occurs when a model is too simple to capture the underlying patterns in the data.

As a result, it performs poorly on both the training data and unseen test data.

### Characteristics

- High training error.
- High validation/testing error.
- Poor predictions.
- Model has **high bias**.

### Example

Trying to fit a straight line through data that follows a curved relationship.

```text
Training Accuracy : Low
Testing Accuracy  : Low
```

### Causes

- Model is too simple.
- Not enough training epochs.
- Too few features.
- Excessive regularization.

### Solutions

- Increase model complexity.
- Train for more epochs.
- Add more relevant features.
- Reduce regularization.

---

# Good Fit

A well-trained model learns the important patterns without memorizing the training data.

### Characteristics

- Low training error.
- Low testing error.
- Good generalization.
- Balanced bias and variance.

```text
Training Accuracy : High
Testing Accuracy  : High
```

---

# Overfitting

**Overfitting** occurs when a model learns not only the underlying patterns but also the noise and random fluctuations in the training data.

It performs extremely well on the training data but poorly on new, unseen data.

### Characteristics

- Very low training error.
- High validation/testing error.
- Poor generalization.
- Model has **high variance**.

### Example

A model memorizes every training sample instead of learning general patterns.

```text
Training Accuracy : Very High
Testing Accuracy  : Low
```

---

# Causes of Overfitting

- Model is too complex.
- Too many parameters.
- Small training dataset.
- Training for too many epochs.
- Noise in the training data.

---

# Solutions to Overfitting

- Collect more training data.
- Data augmentation.
- Early stopping.
- Dropout.
- L1/L2 regularization.
- Reduce model complexity.
- Cross-validation.

---

# Underfitting vs Overfitting

| Feature | Underfitting | Overfitting |
|----------|--------------|-------------|
| Training Accuracy | Low | Very High |
| Testing Accuracy | Low | Low |
| Training Error | High | Very Low |
| Testing Error | High | High |
| Bias | High | Low |
| Variance | Low | High |
| Model Complexity | Too Simple | Too Complex |

---

# Bias-Variance Tradeoff

Model performance is often explained using the **bias-variance tradeoff**.

- **High Bias** → Underfitting
- **High Variance** → Overfitting

The objective is to find a balance between bias and variance so the model generalizes well.

---

# Regularization

**Regularization** is a set of techniques used to reduce overfitting by preventing the model from becoming overly complex.

It encourages the model to learn simpler and more general patterns instead of memorizing the training data.

---

# L1 Regularization (Lasso)

L1 Regularization adds the absolute values of the weights to the loss function.

### Formula

$$
L_{\text{new}} = L + \lambda \sum |w|
$$

Where:

- $L$ = Original loss
- $\lambda$ = Regularization strength
- $w$ = Model weights

### Characteristics

- Encourages sparse models.
- Some weights become exactly zero.
- Performs automatic feature selection.

### Advantages

- Reduces overfitting.
- Can eliminate unimportant features.

---

# L2 Regularization (Ridge / Weight Decay)

L2 Regularization adds the squared values of the weights to the loss.

### Formula

$$
L_{\text{new}} = L + \lambda \sum w^2
$$

### Characteristics

- Reduces large weights.
- Keeps all features but limits their influence.
- More commonly used in deep learning than L1.

### Advantages

- Improves generalization.
- Produces smoother models.
- Reduces overfitting.

---

# Dropout

**Dropout** is one of the most popular regularization techniques in deep learning.

During training, it randomly disables (drops) a percentage of neurons in each iteration.

Example:

```text
Before Dropout

●──●──●──●
│  │  │  │
●──●──●──●

After Dropout

●──X──●──X
│     │
●──●──X──●
```

`X` represents neurons that are temporarily turned off during training.

This forces the network to rely on different neurons instead of becoming dependent on a few specific ones.

### Advantages

- Reduces overfitting.
- Improves generalization.
- Easy to implement.

---

# Early Stopping

Sometimes a model starts to overfit after many training epochs.

**Early Stopping** monitors the validation loss during training.

If the validation loss stops improving for several consecutive epochs, training is stopped automatically.

```text
Training Loss
│
│\
│ \
│  \
│   \
│    \
└──────────► Epoch

Validation Loss
│\
│ \
│  \
│   \
│    \__
│       \__
└──────────► Epoch
        ↑
   Stop Training
```

Early stopping prevents the model from memorizing the training data.

---

# Data Augmentation

Data augmentation artificially increases the size of the training dataset by creating modified versions of existing samples.

For images, common techniques include:

- Rotation
- Flipping
- Cropping
- Zooming
- Brightness adjustment
- Noise addition

Benefits:

- More training data.
- Better generalization.
- Reduced overfitting.

---

# Common Regularization Techniques

| Technique | Purpose |
|-----------|---------|
| L1 Regularization | Feature selection by shrinking some weights to zero |
| L2 Regularization | Reduces large weights while keeping all features |
| Dropout | Randomly disables neurons during training |
| Early Stopping | Stops training before overfitting occurs |
| Data Augmentation | Increases dataset diversity |
| Cross-Validation | Evaluates model robustness across multiple data splits |

---

# Summary

| Situation | Training Accuracy | Testing Accuracy |
|-----------|-------------------|------------------|
| Underfitting | Low | Low |
| Good Fit | High | High |
| Overfitting | Very High | Low |

---

# Key Takeaways

- **Underfitting** occurs when a model is too simple and fails to learn the underlying patterns in the data.
- **Overfitting** occurs when a model memorizes the training data, including noise, and performs poorly on unseen data.
- A **good model** achieves high performance on both training and testing data by generalizing well.
- **Regularization** techniques help reduce overfitting and improve model generalization.
- **L1 Regularization** encourages sparse models by driving some weights to zero.
- **L2 Regularization** (Weight Decay) penalizes large weights and is widely used in deep learning.
- **Dropout**, **Early Stopping**, and **Data Augmentation** are effective techniques for preventing overfitting in neural networks.
