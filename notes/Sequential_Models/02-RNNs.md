# Recurrent Neural Networks (RNNs) and Their Limitations

## What is a Recurrent Neural Network (RNN)?

A **Recurrent Neural Network (RNN)** is a type of neural network designed to process **sequential data**, where the order of inputs matters. Unlike traditional feedforward neural networks, an RNN has a **memory** that allows it to retain information from previous inputs and use it when processing the current input.

This makes RNNs well-suited for tasks involving sequences such as text, speech, time series, and video.

---

# Why Do We Need RNNs?

Traditional feedforward neural networks process each input independently and cannot remember previous inputs.

For example, in the sentence:

```text
The cat sat on the ______.
```

To correctly predict the next word (**mat**), the model must remember the previous words. A feedforward network cannot do this because it has no memory.

An RNN solves this problem by maintaining a **hidden state** that carries information from one time step to the next.

---

# How an RNN Works

An RNN processes one element of a sequence at a time while updating its hidden state.

```text
x₁ → [RNN] → h₁
             ↓
x₂ → [RNN] → h₂
             ↓
x₃ → [RNN] → h₃
             ↓
x₄ → [RNN] → h₄
```

Where:

- **xₜ** = input at time step *t*
- **hₜ** = hidden state (memory) at time step *t*

The hidden state stores information from previous time steps and passes it to the next step.

---

# Hidden State

At each time step, the hidden state is updated using both the current input and the previous hidden state.

The hidden state is computed as:

$$
h_t = f(W_h h_{t-1} + W_x x_t + b)
$$

Where:

- $$h_t$$ = current hidden state
- $$h_{t-1}$$ = previous hidden state
- $$x_t$$ = current input
- $$W_h$$ = recurrent weight matrix
- $$W_x$$ = input weight matrix
- $$b$$ = bias
- $$f$$ = activation function (usually **tanh**)

The output at each time step can then be computed from the hidden state.

---

# Unrolled RNN

Although an RNN is a single neural network, it is often represented as an **unrolled network** to illustrate how it processes sequences.

```text
        h₀
         │
x₁ → [RNN] → h₁
             │
x₂ → [RNN] → h₂
             │
x₃ → [RNN] → h₃
             │
x₄ → [RNN] → h₄
```

The same network (with the same weights) is reused at every time step.

---

# Example

Consider the sentence:

```text
I love deep learning
```

Processing steps:

### Step 1

```text
Input:
I

Memory:
"I"
```

### Step 2

```text
Input:
love

Memory:
"I love"
```

### Step 3

```text
Input:
deep

Memory:
"I love deep"
```

### Step 4

```text
Input:
learning

Memory:
"I love deep learning"
```

The hidden state continually updates as more words are processed.

---

# Advantages of RNNs

- Designed for sequential data.
- Maintains memory through hidden states.
- Can process sequences of varying lengths.
- Shares the same parameters across all time steps, reducing the number of learnable parameters.
- Effective for modeling short-term dependencies.

---

# Limitations of RNNs

Despite their usefulness, vanilla RNNs suffer from several major limitations.

---

## 1. Vanishing Gradient Problem

During training, RNNs use **Backpropagation Through Time (BPTT)** to update their weights.

As gradients are propagated backward through many time steps, they can become extremely small.

Example:

```text
0.9 × 0.9 × 0.9 × 0.9 × ...
```

Eventually,

```text
≈ 0
```

As a result:

- Weight updates become negligible.
- Early layers learn very slowly.
- The model struggles to remember information from the distant past.

This is known as the **vanishing gradient problem**.

---

## 2. Exploding Gradient Problem

In some cases, gradients become extremely large during backpropagation.

Example:

```text
2 × 2 × 2 × 2 × ...
```

This causes:

- Unstable training
- Extremely large weight updates
- Numerical overflow
- Failure to converge

A common solution is **gradient clipping**, which limits the maximum gradient value during training.

---

## 3. Short-Term Memory

Standard RNNs can only remember information from a few previous time steps.

For example:

```text
The weather was extremely cold yesterday because ...
...
...
...
Today it snowed.
```

By the time the network reaches the word **snowed**, it may have forgotten the earlier context (**cold yesterday**).

This makes vanilla RNNs ineffective for learning long-term dependencies.

---

## 4. Sequential Computation

RNNs process one time step after another.

```text
x₁ → x₂ → x₃ → x₄
```

Since each step depends on the previous hidden state, computations cannot be fully parallelized.

As a result:

- Training is slower.
- GPUs cannot be utilized as efficiently as with parallel architectures.

---

## 5. Difficulty Learning Long Sequences

As sequence length increases:

- Memory becomes weaker.
- Important information is gradually forgotten.
- Training becomes more difficult.

This limits the performance of vanilla RNNs on long documents, long conversations, and lengthy time-series data.

---

# Solutions to RNN Limitations

Several improved architectures were developed to overcome these challenges.

| Problem | Solution |
|----------|----------|
| Vanishing gradients | LSTM, GRU |
| Exploding gradients | Gradient clipping |
| Short-term memory | LSTM, GRU |
| Slow sequential computation | Transformers |

---

# Applications of RNNs

Although modern architectures often outperform them, RNNs are still useful for:

- Language modeling
- Text generation
- Speech recognition
- Time-series forecasting
- Music generation
- Handwriting recognition

---

# Key Takeaways

- **RNNs** are neural networks designed for processing sequential data by maintaining a **hidden state** (memory).
- The hidden state is updated at each time step using both the current input and the previous hidden state.
- RNNs are effective for modeling **short-term dependencies** in sequences.
- Their main limitations are the **vanishing gradient problem**, **exploding gradient problem**, **short-term memory**, and **slow sequential computation**.
- These limitations motivated the development of **LSTMs**, **GRUs**, and eventually **Transformers**, which are better suited for learning long-range dependencies.
