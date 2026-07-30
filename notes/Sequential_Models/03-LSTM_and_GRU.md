# Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)

## Introduction

Standard **Recurrent Neural Networks (RNNs)** struggle to learn long-term dependencies because of problems such as the **vanishing gradient** and **exploding gradient**. As sequences become longer, information from earlier time steps is gradually lost, making it difficult for the network to remember important context.

To overcome these limitations, two improved recurrent architectures were introduced:

- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**

Both models use **gating mechanisms** to control the flow of information, allowing them to remember relevant information for much longer than a standard RNN.

---

# Long Short-Term Memory (LSTM)

## What is an LSTM?

A **Long Short-Term Memory (LSTM)** network is an advanced type of RNN specifically designed to capture **long-term dependencies** in sequential data.

Unlike a standard RNN, an LSTM maintains two states:

- **Hidden State ($$h_t$$):** Represents the output at the current time step.
- **Cell State ($$C_t$$):** Acts as long-term memory, carrying important information through the sequence.

The cell state allows information to flow with minimal changes, helping preserve important information over many time steps.

---

# LSTM Architecture

```text
                 Cell State (Long-Term Memory)
────────────────────────────────────────────────────►

          ↓ Forget Gate

          ↓ Input Gate

          ↓ Candidate Memory

          ↓ Output Gate

                 Hidden State
```

Instead of replacing its memory at every step, an LSTM carefully decides:

- What information to forget.
- What new information to store.
- What information to output.

---

# Components of an LSTM

## 1. Forget Gate

The **forget gate** determines which information from the previous cell state should be discarded.

Formula:

$$
f_t=\sigma(W_f[h_{t-1},x_t]+b_f)
$$

Where:

- $$\sigma$$ is the sigmoid activation function.
- Values close to **1** mean **keep the information**.
- Values close to **0** mean **discard the information**.

---

## 2. Input Gate

The input gate determines which new information should be added to the cell state.

### Step 1: Compute the input gate

$$
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
$$

### Step 2: Generate candidate memory

$$
\tilde{C}_t=\tanh(W_c[h_{t-1},x_t]+b_c)
$$

### Step 3: Update the cell state

$$
C_t=f_t \odot C_{t-1}+i_t \odot \tilde{C}_t
$$

Where $$\odot$$ denotes element-wise multiplication.

---

## 3. Output Gate

The output gate determines what information from the cell state becomes the hidden state.

Output gate:

$$
o_t=\sigma(W_o[h_{t-1},x_t]+b_o)
$$

Hidden state:

$$
h_t=o_t \odot \tanh(C_t)
$$

The hidden state is then passed to the next time step and can also be used to make predictions.

---

# LSTM Memory Flow

```text
Previous Cell State
        │
        ▼
  Forget Gate
        │
        ▼
   Input Gate
        │
        ▼
 Updated Cell State
        │
        ▼
   Output Gate
        │
        ▼
   Hidden State
```

Each gate controls a specific part of the information flow, allowing the network to preserve important information over long sequences.

---

# Advantages of LSTM

- Learns long-term dependencies effectively.
- Solves the vanishing gradient problem.
- Maintains separate short-term and long-term memory.
- Performs well on long sequences.
- Widely used in NLP and speech applications.

---

# Disadvantages of LSTM

- Large number of trainable parameters.
- Computationally expensive.
- Slower training.
- Higher memory usage.

---

# Gated Recurrent Unit (GRU)

## What is a GRU?

A **Gated Recurrent Unit (GRU)** is a simplified version of the LSTM.

Unlike LSTM, a GRU:

- Does **not** use a separate cell state.
- Combines the hidden state and memory into a single representation.
- Uses only **two gates** instead of three.

Because of its simpler design, GRUs are generally faster to train while achieving performance comparable to LSTMs.

---

# GRU Architecture

```text
Input
  │
  ▼
Update Gate
  │
  ▼
Reset Gate
  │
  ▼
Candidate Hidden State
  │
  ▼
New Hidden State
```

---

# Components of a GRU

## 1. Update Gate

The update gate decides how much of the previous hidden state should be retained.

Formula:

$$
z_t=\sigma(W_z[h_{t-1},x_t])
$$

If the update gate outputs a value close to **1**, most of the previous information is preserved.

---

## 2. Reset Gate

The reset gate determines how much previous information should be ignored when computing the candidate hidden state.

Formula:

$$
r_t=\sigma(W_r[h_{t-1},x_t])
$$

A small reset gate value allows the model to ignore irrelevant past information.

---

## 3. Candidate Hidden State

The candidate hidden state is computed using the reset gate.

$$
\tilde{h}_t=\tanh(W[r_t \odot h_{t-1},x_t])
$$

---

## 4. Final Hidden State

The final hidden state combines the previous hidden state and the candidate hidden state.

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t
$$

The update gate determines how much information comes from the previous hidden state versus the candidate hidden state.

---

# Advantages of GRU

- Simpler architecture than LSTM.
- Fewer trainable parameters.
- Faster training.
- Lower memory consumption.
- Performs well on many sequential tasks.

---

# Disadvantages of GRU

- Slightly less expressive than LSTM.
- May underperform LSTMs on very long or highly complex sequences.

---

# LSTM vs GRU

| Feature | LSTM | GRU |
|----------|------|-----|
| Cell State | ✅ Yes | ❌ No |
| Hidden State | ✅ Yes | ✅ Yes |
| Number of Gates | 3 | 2 |
| Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Memory Usage | Higher | Lower |
| Long-Term Memory | Excellent | Very Good |
| Complexity | Higher | Lower |

---

# When to Use LSTM or GRU

### Use LSTM when:

- Long-term dependencies are critical.
- The dataset is large.
- Maximum accuracy is more important than training speed.
- Working on complex NLP or speech tasks.

### Use GRU when:

- Faster training is required.
- Memory or computational resources are limited.
- The dataset is relatively small or medium-sized.
- Performance similar to LSTM is sufficient.

---

# TensorFlow/Keras Examples

## LSTM

```python
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    Input(shape=(100, 20)),
    LSTM(64),
    Dense(1)
])
```

---

## GRU

```python
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

model = Sequential([
    Input(shape=(100, 20)),
    GRU(64),
    Dense(1)
])
```

---

# Summary Comparison

| Property | RNN | LSTM | GRU |
|----------|-----|------|-----|
| Handles Sequential Data | ✅ | ✅ | ✅ |
| Long-Term Memory | ❌ | ✅ | ✅ |
| Vanishing Gradient | Poor | Excellent | Excellent |
| Cell State | ❌ | ✅ | ❌ |
| Gates | None | 3 | 2 |
| Parameters | Few | Most | Moderate |
| Training Speed | Fast | Slow | Faster |
| Memory Usage | Low | High | Medium |

---

# Key Takeaways

- **LSTM** and **GRU** are advanced recurrent neural networks designed to overcome the limitations of vanilla RNNs.
- **LSTMs** use a **cell state** and **three gates** (forget, input, and output) to preserve long-term information.
- **GRUs** simplify the LSTM architecture by removing the cell state and using only **two gates** (update and reset).
- **LSTMs** generally provide better performance on very long sequences, while **GRUs** are faster and require fewer computational resources.
- Both architectures remain important for sequential data tasks, although **Transformers** have become the dominant architecture for many modern NLP applications.
